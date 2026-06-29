import { get, set } from 'idb-keyval';
import { nanoid } from 'nanoid';
// @ts-ignore
import BM25Worker from '../workers/bm25.worker?worker';

export interface Memory {
  id: string;
  text: string;
  timestamp: number;
  isCore?: boolean;
  turnId?: string;
}

class RAGService {
  // Biến ảo để tương thích ngược với code cũ trong Setting và Gameplay
  public downloadProgress = 100;
  public downloadStatus: 'idle' | 'downloading' | 'success' | 'error' = 'success';
  public currentModel = 'ai-native-bm25-rag';
  
  public get getDownloadStatus() { return this.downloadStatus; }
  public get getDownloadProgress() { return this.downloadProgress; }

  // Các hàm ảo để không bị lỗi code cũ gọi đến
  public async checkModelCached() { return true; }
  public async forceCheckModelCached() { return true; }
  public async preloadModelFromSettings(onProgress?: (progress: number, status: string) => void) {
    if (onProgress) onProgress(100, 'success');
  }
  public async init(modelName?: string, onProgress?: (progress: number, status: string) => void) {
    if (onProgress) onProgress(100, 'success');
  }

  private getDBKey(saveId: string) {
    const cleanId = saveId || 'temp_session';
    return `rag_memories_${cleanId}`;
  }

  public async getMemories(saveId: string): Promise<Memory[]> {
    if (!saveId) return [];
    const data = await get(this.getDBKey(saveId));
    return data || [];
  }

  // Quản lý Web Worker (Đa luồng chống giật UI)
  private worker: Worker | null = null;
  private messageCounter = 0;
  private resolves = new Map<number, (value: any) => void>();
  private rejects = new Map<number, (reason?: any) => void>();
  private syncedSaveIds = new Set<string>();

  private initWorker() {
    if (!this.worker) {
      this.worker = new BM25Worker();
      this.worker.onmessage = (e) => {
        const { type, messageId, payload } = e.data;
        if (type === 'ERROR') {
          const reject = this.rejects.get(messageId);
          if (reject) reject(new Error(payload));
        } else {
          const resolve = this.resolves.get(messageId);
          if (resolve) resolve(payload);
        }
        this.resolves.delete(messageId);
        this.rejects.delete(messageId);
      };
    }
  }

  private runWorkerTask(type: string, payload: any): Promise<any> {
    this.initWorker();
    return new Promise((resolve, reject) => {
      const messageId = ++this.messageCounter;
      this.resolves.set(messageId, resolve);
      this.rejects.set(messageId, reject);
      this.worker!.postMessage({ type, messageId, payload });
    });
  }

  private async syncWorkerCache(saveId: string, force: boolean = false) {
    if (this.syncedSaveIds.has(saveId) && !force) return;
    const memories = await this.getMemories(saveId);
    const standardMemories = memories.filter(m => !m.isCore);
    // Gửi dữ liệu rút gọn (chỉ id và text) sang Worker để build Cache & Inverted Index
    await this.runWorkerTask('SYNC_CACHE', { 
      saveId, 
      memories: standardMemories.map(m => ({ id: m.id, text: m.text })) 
    });
    this.syncedSaveIds.add(saveId);
  }

  public async addMemory(saveId: string, text: string, isCore: boolean = false, proxy?: any, turnId?: string): Promise<Memory> {
    const memory: Memory = {
      id: nanoid(),
      text,
      timestamp: Date.now(),
      isCore,
      turnId
    };

    const targetId = saveId || 'temp_session';
    const memories = await this.getMemories(targetId);
    memories.push(memory);
    await set(this.getDBKey(targetId), memories);

    // Cập nhật lại Cache trên Worker
    await this.syncWorkerCache(targetId, true);

    return memory;
  }

  public async deleteMemoriesByTurnId(saveId: string, turnId: string) {
    if (!turnId) return;
    const memories = await this.getMemories(saveId);
    const updated = memories.filter(m => m.turnId !== turnId);
    if (updated.length !== memories.length) {
      await set(this.getDBKey(saveId), updated);
      await this.syncWorkerCache(saveId, true);
    }
  }

  // Thuật toán BM25 trên Web Worker
  public async searchMemory(saveId: string, query: string, topK: number = 3, threshold: number = 0.1, proxy?: any): Promise<{core: Memory[], standard: Memory[]}> {
    const memories = await this.getMemories(saveId);
    if (memories.length === 0) return { core: [], standard: [] };

    const coreMemories = memories.filter(m => m.isCore);
    
    // Đảm bảo Worker đã nạp dữ liệu
    await this.syncWorkerCache(saveId);

    // Ủy quyền tính toán cho luồng ngầm (Web Worker)
    const searchResults: { id: string, score: number }[] = await this.runWorkerTask('SEARCH', {
      saveId,
      query,
      topK,
      threshold
    });

    // Lookup lại full Memory object từ DB nội bộ
    const standardResultMemories = searchResults
      .map(res => memories.find(m => m.id === res.id))
      .filter(m => m !== undefined) as Memory[];

    return {
      core: coreMemories,
      standard: standardResultMemories
    };
  }

  public async retrieveContext(saveId: string, query: string, maxTokens: number = 1000, proxy?: any): Promise<string> {
    const { core, standard } = await this.searchMemory(saveId, query, 3, 0.1, proxy);
    
    if (core.length === 0 && standard.length === 0) return "";

    let context = "--- NHỮNG KÝ ỨC VÀ KIẾN THỨC BẠN NHỚ LẠI ĐƯỢC ---\n\n";
    if (core.length > 0) {
      context += "[KÝ ỨC CỐT LÕI (Không bao giờ quên)]:\n";
      core.forEach(m => context += `- ${m.text}\n`);
      context += "\n";
    }

    if (standard.length > 0) {
      context += "[KÝ ỨC LIÊN QUAN ĐẾN HOÀN CẢNH HIỆN TẠI]:\n";
      standard.forEach(m => context += `- ${m.text}\n`);
    }

    const approximateCharLimit = maxTokens * 4;
    if (context.length > approximateCharLimit) {
      return context.substring(0, approximateCharLimit) + "...";
    }
    return context + "\n------------------------------------------------";
  }

  public async deleteMemory(saveId: string, memoryId: string) {
    const memories = await this.getMemories(saveId);
    const updated = memories.filter(m => m.id !== memoryId);
    await set(this.getDBKey(saveId), updated);
    await this.syncWorkerCache(saveId, true);
  }

  public async clearAllMemories(saveId: string) {
    await set(this.getDBKey(saveId), []);
    await this.syncWorkerCache(saveId, true);
  }

  public async clearMemories(saveId: string) {
    await this.clearAllMemories(saveId);
  }

  public async setMemories(saveId: string, memories: Memory[]) {
    await set(this.getDBKey(saveId), memories || []);
    await this.syncWorkerCache(saveId, true);
  }

  public get isFallback() { return false; }
}

export const ragService = new RAGService();


