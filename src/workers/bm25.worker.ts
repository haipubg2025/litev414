// BM25 Worker: Xử lý tìm kiếm toàn văn trên luồng chạy ngầm để không block UI (Drop Frame)
// Hỗ trợ Hashing Token ID và Lọc Stop Words Tiếng Việt.

// 1. Bộ lọc Stop Words Tiếng Việt: Loại bỏ các từ nối vô nghĩa giúp giảm size Cache 30% - 40%
const STOP_WORDS = new Set([
  'bị', 'bởi', 'cả', 'các', 'cái', 'cần', 'càng', 'chỉ', 'chiếc', 'cho', 'chứ', 'chưa', 
  'chuyện', 'có', 'có_thể', 'cứ', 'của', 'cùng', 'cũng', 'đã', 'đang', 'đây', 'để', 
  'đến_nỗi', 'đều', 'điều', 'do', 'đó', 'được', 'dưới', 'gì', 'khi', 'không', 'là', 
  'lại', 'lên', 'lúc', 'mà', 'mỗi', 'một_cách', 'này', 'nên', 'nếu', 'ngay', 'nhiều', 
  'như', 'nhưng', 'những', 'nơi', 'nữa', 'phải', 'qua', 'ra', 'rằng', 'rất', 'rồi', 
  'sau', 'sẽ', 'so', 'sự', 'tại', 'theo', 'thì', 'trên', 'trong', 'trước', 'từ', 
  'từng', 'và', 'vẫn', 'vào', 'vậy', 'vì', 'việc', 'với', 'vừa'
]);

// 2. Token ID Hashing: Chuyển string thành Integer giúp CPU quét siêu nhanh và tiết kiệm RAM
let nextTokenId = 1;
const tokenToId = new Map<string, number>();

function getTokenId(token: string, addIfMissing: boolean = true): number | undefined {
  let id = tokenToId.get(token);
  if (id !== undefined) return id;
  if (!addIfMissing) return undefined;
  id = nextTokenId++;
  tokenToId.set(token, id);
  return id;
}

function tokenize(text: string, addIfMissing: boolean = true): number[] {
  // Tách từ, bỏ ký tự đặc biệt, lọc Stop words
  const words = text.toLowerCase()
    .replace(/[^\p{L}\p{N}\s_]/gu, ' ')
    .split(/\s+/)
    .filter(w => w.length > 0 && !STOP_WORDS.has(w));
    
  const tokens: number[] = [];
  
  for (const w of words) {
    const id = getTokenId(w, addIfMissing);
    if (id !== undefined) tokens.push(id);
  }

  // Tạo Bigram (cụm 2 từ liền kề)
  for (let i = 0; i < words.length - 1; i++) {
    const bigram = `${words[i]}_${words[i+1]}`;
    const id = getTokenId(bigram, addIfMissing);
    if (id !== undefined) tokens.push(id);
  }
  
  return tokens;
}

interface CacheData {
  standardMemories: { id: string, text: string }[];
  docTokensList: number[][]; 
  df: Record<number, number>;
  totalDocLength: number;
  avgDocLength: number;
}

const caches: Record<string, CacheData> = {};

// 3. Lắng nghe và xử lý đa luồng (Web Worker)
self.onmessage = (e: MessageEvent) => {
  const { type, messageId, payload } = e.data;

  try {
    if (type === 'SYNC_CACHE') {
      const { saveId, memories } = payload;
      
      let totalDocLength = 0;
      const docTokensList: number[][] = [];
      const df: Record<number, number> = {};

      // Xây dựng Inverted Index bằng Integer Hashing
      for (const m of memories) {
        const tokens = tokenize(m.text, true);
        docTokensList.push(tokens);
        totalDocLength += tokens.length;
        
        const uniqueTokens = new Set(tokens);
        for (const t of uniqueTokens) {
          df[t] = (df[t] || 0) + 1;
        }
      }

      const N = memories.length;
      const avgDocLength = totalDocLength / Math.max(N, 1);

      caches[saveId] = {
        standardMemories: memories,
        docTokensList,
        df,
        totalDocLength,
        avgDocLength
      };

      self.postMessage({ type: 'SYNC_CACHE_SUCCESS', messageId });
    } 
    else if (type === 'SEARCH') {
      const { saveId, query, topK, threshold } = payload;
      const cache = caches[saveId];
      
      if (!cache || cache.standardMemories.length === 0) {
        self.postMessage({ type: 'SEARCH_SUCCESS', messageId, payload: [] });
        return;
      }

      const { standardMemories, docTokensList, df, avgDocLength } = cache;
      const N = standardMemories.length;

      // Tokenize truy vấn (không add token mới vào từ điển)
      const qTokenIds = tokenize(query, false).filter(id => id !== undefined);
      
      if (qTokenIds.length === 0) {
        // Fallback trả về danh sách ngẫu nhiên/cũ nhất nếu không có từ khóa
        const results = standardMemories.slice(0, topK).map(m => ({ id: m.id, score: 0 }));
        self.postMessage({ type: 'SEARCH_SUCCESS', messageId, payload: results });
        return;
      }

      const k1 = 1.5;
      const b = 0.75;
      const results = [];
      const queryLower = query.toLowerCase().trim();

      for (let i = 0; i < standardMemories.length; i++) {
        const memory = standardMemories[i];
        const tokens = docTokensList[i];
        const docLength = tokens.length;
        let score = 0;

        const tf: Record<number, number> = {};
        for (const t of tokens) {
          tf[t] = (tf[t] || 0) + 1;
        }

        // Tính BM25 Score dựa trên Integer Array
        for (const qId of qTokenIds) {
          let fq = tf[qId] || 0;

          if (fq > 0) {
            const docFreq = df[qId] || 1;
            const idf = Math.log(1 + (N - docFreq + 0.5) / (docFreq + 0.5));
            const termScore = idf * (fq * (k1 + 1)) / (fq + k1 * (1 - b + b * (docLength / avgDocLength)));
            score += termScore;
          }
        }
        
        // Boost điểm nếu chứa cụm chuỗi gốc chính xác
        if (memory.text.toLowerCase().includes(queryLower)) {
           score += 5.0; 
        }

        if (score >= threshold) {
          results.push({ id: memory.id, score });
        }
      }

      results.sort((a, b) => b.score - a.score);
      const topResults = results.slice(0, topK);
      
      // Chỉ trả về ID và Score (siêu nhẹ băng thông)
      self.postMessage({ type: 'SEARCH_SUCCESS', messageId, payload: topResults });
    }
  } catch (error: any) {
    self.postMessage({ type: 'ERROR', messageId, payload: error.message });
  }
};
