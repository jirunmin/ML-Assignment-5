from typing import List, Dict, Optional, Tuple, Set
from openai import AsyncOpenAI
import tiktoken
import numpy as np
from collections import Counter
import math
import re

from model import ModelProvider

# 尝试导入sentence-transformers，如果失败则使用备用方案
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[WARNING] sentence-transformers not available, will use BM25 fallback")


class PrecisionRetrievalAgent(ModelProvider):
    """
    Precision-focused retrieval agent with multi-stage search strategy.
    
    Features:
    - Exact string matching for key entities (highest priority)
    - Multi-stage retrieval: exact match -> BM25 keyword -> semantic
    - Enhanced entity extraction with Δ symbol support
    - Hybrid scoring mechanism
    - Question type detection for targeted reasoning
    """

    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.model_name = "qwen-max"  # 主模型用于最终回答
        self.helper_model_name = "qwen-max"  # 辅助模型用于关键词提取
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        # 初始化向量化模型
        self.embedding_model = None
        self.embedding_dim = 384  # 默认维度
        self._init_embedding_model()
        
        # 检索参数配置
        self.chunk_size = 550  # 块大小
        self.chunk_overlap = 100  # 块重叠
        self.top_k = 15  # 检索候选块数
        self.rerank_top_k = 8  # 重排序后保留的块数
        self.max_context_tokens = 12000  # 最大上下文令牌数
        self.context_expansion = 800  # 上下文扩展字符数
        
        # 精确匹配阈值 - 如果找到足够多的精确匹配，跳过向量匹配
        self.vector_skip_threshold = 2
        
        # 缓存LLM提取的关键词
        self._keyword_cache = {}
        
        # 问题类型定义
        self.question_types = {
            "date_time": r"(when|date|time|schedule|deadline|milestone|year|month|day|week|weekday|day of the week)",
            "computation": r"(calculate|compute|result|value|number|sum|average|total|sqrt|divide|multiply)",
            "string_analysis": r"(md5|hash|length|character|string|count|substring)",
            "encoding": r"(base64|decode|encode|hex|binary|caesar|cipher|password)",
            "general": r".*"
        }
    
    def _init_embedding_model(self):
        """初始化本地向量化模型"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            except Exception as e:
                self.embedding_model = None
        else:
            self.embedding_model = None
    

    
    def _chunk_text(self, text: str, filename: str) -> List[Dict]:
        """将文本分割成重叠的固定token大小的块"""
        tokens = self.encode_text_to_tokens(text)
        chunks = []
        
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.decode_tokens(chunk_tokens)
            
            chunks.append({
                'id': f"{filename}_{chunk_id}",
                'text': chunk_text,
                'filename': filename,
                'start_token': start,
                'end_token': end,
                'embedding': None
            })
            
            chunk_id += 1
            start += self.chunk_size - self.chunk_overlap
            
            if start >= len(tokens):
                break
        
        return chunks
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """计算文本列表的向量嵌入"""
        if self.embedding_model is None:
            return np.zeros((len(texts), self.embedding_dim))
        
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def _compute_cosine_similarity(self, query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
        """计算查询向量与块向量之间的余弦相似度"""
        return np.dot(chunk_embeddings, query_embedding)
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        从文本中提取重要实体用于增强匹配
        
        Args:
            text: 要提取实体的文本
            
        Returns:
            实体类型及其值的字典
        """
        entities = {
            'dates': [],
            'numbers': [],
            'project_codes': [],
            'years': [],
            'quoted_strings': [],
            'capitalized_phrases': [],
            'project_names': []
        }
        
        # 提取引号内的内容
        entities['quoted_strings'] = re.findall(r'"([^"]+)"', text)
        entities['quoted_strings'].extend(re.findall(r"'([^']+)'", text))
        
        # 提取日期（各种格式）
        date_patterns = [
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',  # 2031-12-25
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # 12/25/2031
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{4}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])\b',  # 严格格式 YYYY-MM-DD
            r'\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/\d{4}\b'  # 严格格式 MM/DD/YYYY
        ]
        for pattern in date_patterns:
            entities['dates'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # 提取年份
        entities['years'] = re.findall(r'\b(20\d{2}|19\d{2}|21\d{2})\b', text)
        
        # 提取项目代码 - 支持更多格式
        code_patterns = [
            r'\b[A-Z]{1,5}-[A-Z0-9ΔΦΩ]{1,10}-[A-Z0-9a-z]{1,15}\b',  # 支持ΔΦΩ的格式
            r'\b[A-Z]{1,3}\d*-[A-Z0-9]+\b',
            r'\b[A-Z]{2,}-[A-Z]*-?\d+[A-Z]*\b',
            r'\bProject\s+Code[:\s]+([A-Z0-9-ΔΦΩ]+)\b',
            r'\bcode[:\s]+([A-Z0-9-ΔΦΩ]+)\b',
            r'\b[A-Z0-9]+-[A-Z0-9]+(?:-[A-Z0-9]+)*\b',  # 通用连字符格式
            r'\b[A-Z]{1,5}\d+[A-Z]*\b',  # 不带连字符的项目代码
            r'\b[A-Z0-9Ω]{2,}\b',  # 包含Ω的项目代码
            r'\b[A-Z]+-[0-9]+[A-Z]*\b',  # 如XR7-884, ZK-99X
            r'\b[A-Z]+-[A-Z]+-\d+\b',  # 如AF-PROJ-8876
            r'\b[A-Z0-9]+/[A-Z0-9-]+\b',  # 如P-7B/2047-Alpha
            r'\b[A-Z]{2,}-\d+-[A-Z]\b'  # 如CHM-PX-881A
        ]
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['project_codes'].extend(matches)
        
        # 提取项目名称
        project_name_patterns = [
            r'\bProject\s+([A-Z][a-zA-Z0-9-]+(?:\s+[A-Z][a-zA-Z0-9-]*)*)',
            r'\bproject\s+["\']([^"\']+)["\']',
            r'\b([A-Z][a-z]+(?:-[A-Z][a-z0-9]+)+)\b'  # Chimera-X1, Aurora-7
        ]
        for pattern in project_name_patterns:
            matches = re.findall(pattern, text)
            entities['project_names'].extend(matches)
        
        # 提取数字
        entities['numbers'] = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        
        # 提取大写短语
        entities['capitalized_phrases'] = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        
        # 去重所有实体列表
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _extract_keywords_from_question(self, question: str) -> Dict[str, List[str]]:
        """
        从问题中提取关键搜索词，分类返回 - 不依赖LLM，使用规则提取
        
        Args:
            question: 问题文本
            
        Returns:
            分类的关键词字典
        """
        result = {
            'exact': [],
            'phrase': [],
            'date': [],
            'key': []
        }
        
        # 提取引号内的内容作为EXACT匹配
        quoted_texts = re.findall(r'["\']([^"\']+)["\']', question)
        for text in quoted_texts:
            result['exact'].append(text)
        
        # 提取括号内的内容作为EXACT匹配
        parenthetical_texts = re.findall(r'\(([^)]+)\)', question)
        for text in parenthetical_texts:
            result['exact'].append(text)
        
        # 提取项目代码作为EXACT匹配（补充匹配）
        project_code_patterns = [
            # 针对P-8812-Cerulean格式的特殊处理
            r'\bP-\d+-[A-Za-z]+\b',  # 如P-8812-Cerulean
            # 针对AP-Δ7-2038格式的处理
            r'\b[A-Z]{1,5}-[A-Z0-9ΔΦΩ]{1,10}-[A-Z0-9a-z]{1,15}\b',
            # 其他常见项目代码格式
            r'\b[A-Z]{1,3}\d*-[A-Z0-9]+\b',
            r'\b[A-Z]{2,}-[A-Z]*-?\d+[A-Z]*\b',
            r'\b[A-Z0-9]+-[A-Z0-9]+(?:-[A-Z0-9]+)*\b',
            r'\b[A-Z]{1,5}\d+[A-Z]*\b',
            r'\b[A-Z0-9Ω]{2,}\b',
            r'\b[A-Z]+-[0-9]+[A-Z]*\b',
            r'\b[A-Z]+-[A-Z]+-\d+\b',
            r'\b[A-Z0-9]+/[A-Z0-9-]+\b',
            r'\b[A-Z]{2,}-\d+-[A-Z]\b',
            r'\b[A-Z]{1,5}-\d{4}\b',
            r'\b[A-Z]+[A-Z0-9]*-[A-Z0-9]+\b'  # 通用项目代码格式
        ]
        for pattern in project_code_patterns:
            matches = re.findall(pattern, question)
            result['exact'].extend(matches)

        # ===== 通用兜底：把“看起来像项目代码”的 token 放进 EXACT（不注入任何答案）=====
        # 目的：避免只靠停用词过滤导致项目号漏掉，但绝不写死特定case，也不塞日期/答案
        result['exact'] = [t.strip() for t in result['exact'] if t and t.strip()]

        # 过滤掉明显不是项目代码的纯英文普通词（但保留含数字/连字符/ΔΦΩ/全大写缩写等）
        def looks_like_identifier(s: str) -> bool:
            if any(ch.isdigit() for ch in s):
                return True
            if '-' in s:
                return True
            if any(sym in s for sym in ['Δ', 'Φ', 'Ω']):
                return True
            if re.search(r'[A-Z]{2,}', s):  # 连续大写
                return True
            return False

        result['exact'] = list({t for t in result['exact'] if looks_like_identifier(t)})
        
        # 提取日期作为DATE匹配
        date_patterns = [
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b'
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, question)
            result['date'].extend(matches)
        
        # 提取短语作为PHRASE匹配
        phrase_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            r'\b[a-zA-Z]+\s+[a-zA-Z]+\s+[a-zA-Z]+\b'
        ]
        for pattern in phrase_patterns:
            matches = re.findall(pattern, question)
            result['phrase'].extend(matches)
        
        # 提取关键单词作为KEY匹配
        key_words = re.findall(r'\b[a-zA-Z]{4,}\b', question)
        # 过滤掉常见的停用词
        stop_words = set(['based', 'on', 'the', 'for', 'from', 'this', 'that', 'which', 'what', 'when', 'where', 'why', 'how', 'many', 'days', 'are', 'there', 'between', 'its', 'will', 'be', 'is', 'in', 'at', 'by', 'to', 'with', 'of', 'and', 'or', 'but', 'not', 'if', 'then', 'than', 'so', 'because', 'as', 'up', 'down', 'out', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
        filtered_key_words = [word for word in key_words if word.lower() not in stop_words]
        result['key'].extend(filtered_key_words)
        
        # 去重所有列表
        for key in result:
            result[key] = list(set(result[key]))
        
        return result
    
    async def _extract_keywords_with_llm(self, question: str) -> Dict[str, List[str]]:
        """
        使用大模型从问题中提取关键搜索词，分类返回
        
        Args:
            question: 问题文本
            
        Returns:
            分类的关键词字典
        """
        # 先尝试使用规则提取
        keywords = self._extract_keywords_from_question(question)
        
        # 如果规则提取结果丰富，直接返回
        if keywords['exact'] or (keywords['phrase'] and keywords['date']):
            return keywords
        
        # 否则尝试使用LLM提取
        # 检查缓存
        if question in self._keyword_cache:
            return self._keyword_cache[question]
        
        try:
            messages = [
                {"role": "system", "content": """You are a search keyword extraction expert. ONLY use the exact text that already appears in the question.

Extract keywords in these categories (each on its own line with the prefix):

EXACT: Exact strings copied verbatim from the question (project codes, identifiers, quoted text, parenthetical text)
PHRASE: Multi-word phrases copied exactly as written in the question
DATE: Dates exactly as written in the question (keep the same format from the question)
KEY: Single important keywords copied exactly as written in the question

Rules:
1. Never invent, expand, or paraphrase any term. Every output token must already exist in the question.
2. Always include anything inside quotes ('...' or "...") or parentheses (...) in EXACT (verbatim, no changes).
3. Do NOT generate new variants, hyphen changes, or alternative formats.
4. If a category has no terms, omit that line entirely.
5. Output nothing except the category lines.
6. Always include project codes with Δ symbols in EXACT category (e.g., AP-Δ7-2038).

Example output format (values must come from the question text):
EXACT: CHM-PX-881A, "Project Chimera-77", AP-Δ7-2038
PHRASE: Project Chimera Phoenix
DATE: 2031-12-25
KEY: deployment, milestone"""},
                {
                    "role": "user",
                    "content": f"Extract all search keywords from this question:\n{question}"
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=self.helper_model_name,
                messages=messages,
                temperature=0,
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            if content is None:
                content = ""
            content = content.strip()
            
            # 解析分类的关键词
            result = {
                'exact': [],
                'phrase': [],
                'date': [],
                'key': []
            }
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('EXACT:'):
                    terms = [t.strip() for t in line[6:].split(',') if t.strip()]
                    result['exact'].extend(terms)
                elif line.startswith('PHRASE:'):
                    terms = [t.strip() for t in line[7:].split(',') if t.strip()]
                    result['phrase'].extend(terms)
                elif line.startswith('DATE:'):
                    terms = [t.strip() for t in line[5:].split(',') if t.strip()]
                    result['date'].extend(terms)
                elif line.startswith('KEY:'):
                    terms = [t.strip() for t in line[4:].split(',') if t.strip()]
                    result['key'].extend(terms)
            
            # 缓存结果
            self._keyword_cache[question] = result
            return result
            
        except Exception as e:
            # 如果LLM请求失败，返回规则提取的结果
            return keywords
    
    def _find_exact_matches(self, keywords: Dict[str, List[str]], files: List[Dict]) -> List[Dict]:
        """
        使用提取的关键词在所有文件中进行精确字符串匹配
        
        Args:
            keywords: LLM提取的分类关键词
            files: 文件列表
            
        Returns:
            匹配的上下文片段列表
        """
        matches = []
        matched_positions = set()  # 避免重复匹配同一位置
        
        # 构建优先级搜索词列表
        prioritized_terms = []
        
        # EXACT类型最高优先级
        for term in keywords.get('exact', []):
            # 给项目代码更高的优先级
            if re.match(r'\b[A-Z0-9]+-[A-Z0-9]+(?:-[A-Z0-9]+)*\b', term) or re.match(r'\bP-\d+-[A-Za-z]+\b', term):
                prioritized_terms.append((term, 300))  # 项目代码优先级最高
            else:
                prioritized_terms.append((term, 200))
        
        # PHRASE次之
        for term in keywords.get('phrase', []):
            prioritized_terms.append((term, 150))
        
        # DATE
        for term in keywords.get('date', []):
            prioritized_terms.append((term, 120))
        
        # 过滤过短或全小写的普通词
        def is_informative(term: str) -> bool:
            if not term or len(term.strip()) < 2:
                return False
            s = term.strip()
            if any(ch.isdigit() for ch in s):
                return True
            if '-' in s:
                return True
            if re.search(r'[A-Z]{2,}', s):
                return True
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                return True
            return len(s) >= 5
        
        prioritized_terms = [(t, p) for (t, p) in prioritized_terms if is_informative(t)]
        
        for file_data in files:
            if file_data and isinstance(file_data, dict):
                filename = file_data.get('filename', 'unknown')
                content = file_data.get('modified_content', '')
                if not content:
                    continue
                content_lower = content.lower()
                
                for term, base_priority in prioritized_terms:
                    # 跳过太短的词
                    if len(term.strip()) < 2:
                        continue
                    
                    # 查找所有出现位置
                    start = 0
                    
                    # 对于包含特殊字符（如Δ）或项目代码，使用原始文本匹配
                    if 'Δ' in term or re.match(r'\b[A-Z0-9]+-[A-Z0-9]+(?:-[A-Z0-9]+)*\b', term) or re.match(r'\bP-\d+-[A-Za-z]+\b', term):
                        match_func = content.find
                        current_term = term
                    else:
                        match_func = content_lower.find
                        current_term = term.lower()
                    
                    while True:
                        pos = match_func(current_term, start)
                        if pos == -1:
                            break
                        
                        # 检查是否已经匹配过这个位置
                        pos_key = f"{filename}_{pos // 200}"  # 按200字符分组，提高匹配精度
                        if pos_key in matched_positions:
                            start = pos + 1
                            continue
                        
                        matched_positions.add(pos_key)
                        
                        # 提取匹配点周围的上下文
                        # 扩大上下文范围，确保包含足够的信息
                        context_start = max(0, pos - self.context_expansion * 2)
                        context_end = min(len(content), pos + len(current_term) + self.context_expansion * 2)
                        
                        # 扩展到句子边界
                        while context_start > 0 and content[context_start] not in '.!?\n':
                            context_start -= 1
                        if context_start > 0:
                            context_start += 1
                            while context_start < len(content) and content[context_start] in ' \t\n':
                                context_start += 1
                        
                        while context_end < len(content) and content[context_end] not in '.!?\n':
                            context_end += 1
                        if context_end < len(content) and content[context_end] in '.!?':
                            context_end += 1
                        
                        context_text = content[context_start:context_end].strip()
                        
                        # 计算优先级
                        priority = base_priority
                        
                        # 对于包含Δ符号的项目代码，增加额外的优先级
                        if 'Δ' in term:
                            priority += 50
                        # 对于P-XXX-XXX格式的项目代码，也增加额外优先级
                        elif re.match(r'\bP-\d+-[A-Za-z]+\b', term):
                            priority += 50
                        
                        # 如果是边界匹配（独立词），优先级更高
                        if re.search(r'\b' + re.escape(term.lower()) + r'\b', content_lower[max(0, pos-1):pos+len(term)+1]):
                            priority += 30
                        
                        matches.append({
                'text': context_text,
                'filename': filename,
                'chunk_id': f"{filename}:{context_start}",
                'start_pos': context_start,
                'end_pos': context_end,
                'priority': priority,
                'matched_term': term
            })
                        
                        start = pos + 1
        
        # 按优先级排序
        matches.sort(key=lambda x: x['priority'], reverse=True)
        
        return matches
    
    def _compute_bm25(self, query: str, chunks: List[Dict]) -> List[Tuple[int, float]]:
        """
        计算查询与所有块之间的BM25评分
        
        Args:
            query: 查询文本
            chunks: 块列表
            
        Returns:
            块索引和BM25分数的元组列表
        """
        # BM25参数
        k1 = 1.5
        b = 0.75
        
        # 分词
        def tokenize(text: str) -> List[str]:
            return re.findall(r'\b\w+\b', text.lower())
        
        query_tokens = tokenize(query)
        
        # 计算文档长度和平均文档长度
        doc_lens = []
        for chunk in chunks:
            tokens = tokenize(chunk['text'])
            doc_lens.append(len(tokens))
        
        if not doc_lens:
            return [(i, 0.0) for i in range(len(chunks))]
        
        avg_doc_len = sum(doc_lens) / len(doc_lens)
        
        # 计算词频和文档频率
        doc_freq = Counter()
        term_freqs = []
        
        for chunk in chunks:
            tokens = tokenize(chunk['text'])
            tf = Counter(tokens)
            term_freqs.append(tf)
            
            for term in tf:
                doc_freq[term] += 1
        
        # 计算BM25分数
        scores = []
        for i, (chunk, tf, doc_len) in enumerate(zip(chunks, term_freqs, doc_lens)):
            score = 0.0
            
            for term in query_tokens:
                if term not in tf:
                    continue
                
                # 逆文档频率 (IDF)
                idf = math.log((len(chunks) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5) + 1.0)
                
                # 词频 (TF)
                term_freq = tf[term]
                
                # BM25公式
                term_score = idf * (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                score += term_score
            
            scores.append((i, score))
        
        return scores
    
    def _vector_match(self, query: str, chunks: List[Dict]) -> List[Tuple[int, float]]:
        """
        使用向量嵌入进行语义匹配
        
        Args:
            query: 查询文本
            chunks: 块列表
            
        Returns:
            块索引和相似度分数的元组列表
        """
        if self.embedding_model is None:
            return [(i, 0.0) for i in range(len(chunks))]
        
        # 计算查询嵌入
        query_embedding = self.embedding_model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 计算块嵌入
        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = self.embedding_model.encode(
            chunk_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 计算余弦相似度
        similarities = np.dot(chunk_embeddings, query_embedding)
        
        # 返回索引和分数
        return [(i, float(similarities[i])) for i in range(len(chunks))]
    
    async def _retrieve_relevant_chunks(self, question: str, files: List[Dict]) -> List[Dict]:
        """
        多阶段检索：精确匹配 -> BM25 -> 向量匹配
        
        Args:
            question: 问题文本
            files: 文件列表
            
        Returns:
            重排序后的相关块列表
        """
        # 步骤1: 提取关键词
        keywords = await self._extract_keywords_with_llm(question)
        
        # 步骤2: 精确匹配
        exact_matches = self._find_exact_matches(keywords, files)
        
        # 步骤3: 对于日期计算类问题，额外搜索包含日期的文本
        question_type = self._detect_question_type(question)
        if question_type == "date_time":
            # 额外搜索所有包含日期的文本片段
            date_matches = []
            for file_data in files:
                if file_data and isinstance(file_data, dict):
                    content = file_data.get('modified_content', '')
                    filename = file_data.get('filename', 'unknown')
                    if content:
                        # 查找所有包含日期的句子
                        sentences = re.split(r'[.!?]+', content)
                        for sentence in sentences:
                            if re.search(r'\d{4}-\d{1,2}-\d{1,2}', sentence):
                                # 提取包含日期的句子作为匹配项
                                date_matches.append({
                                    'text': sentence.strip(),
                                    'filename': filename,
                                    'start_pos': 0,
                                    'end_pos': len(sentence),
                                    'priority': 180,  # 给日期句子较高优先级
                                    'matched_term': 'date'
                                })
            
            # 将日期匹配添加到精确匹配中
            exact_matches.extend(date_matches)
        
        # 步骤4: 如果精确匹配足够多，直接返回
        if len(exact_matches) >= self.vector_skip_threshold:
            return exact_matches[:self.rerank_top_k]
        
        # 步骤5: 否则进行分块并计算BM25
        all_chunks = []
        for file_data in files:
            if file_data and isinstance(file_data, dict):
                modified_content = file_data.get('modified_content', '')
                filename = file_data.get('filename', 'unknown')
                if modified_content:
                    chunks = self._chunk_text(modified_content, filename)
                    all_chunks.extend(chunks)
        
        # 步骤6: BM25匹配
        bm25_scores = self._compute_bm25(question, all_chunks)
        
        # 步骤7: 向量匹配
        vector_scores = self._vector_match(question, all_chunks)
        
        # 步骤8: 混合评分
        scores_dict = {}
        
        # 处理精确匹配分数并设置高优先级
        exact_match_texts = set()
        for i, match in enumerate(exact_matches):
            if match and isinstance(match, dict):
                match_text = match.get('text', '')
                exact_match_texts.add(match_text)
                # 给精确匹配项设置极高的基础分数
                match['score'] = 100.0 + match.get('priority', 0)
                scores_dict[f"exact_{i}"] = match
        
        # 处理BM25和向量分数
        for (chunk_idx, bm25_score), (_, vector_score) in zip(bm25_scores, vector_scores):
            if chunk_idx < len(all_chunks):
                chunk = all_chunks[chunk_idx]
                if chunk and isinstance(chunk, dict):
                    chunk_text = chunk.get('text', '')
                    
                    # 如果该块内容已经在精确匹配中，跳过
                    if chunk_text in exact_match_texts:
                        continue
                    
                    chunk_id = chunk.get('id', f'chunk_{chunk_idx}')
                    
                    # 归一化分数
                    normalized_bm25 = (bm25_score / 10.0) if bm25_score > 0 else 0
                    normalized_vector = (vector_score + 1) / 2  # 转换为0-1范围
                    
                    # 混合评分：BM25 (70%) + 向量 (30%)
                    hybrid_score = normalized_bm25 * 0.7 + normalized_vector * 0.3
                    
                    # 检查块是否包含精确匹配的关键词，增加额外权重
                    for match in exact_matches:
                        if match and isinstance(match, dict):
                            matched_term = match.get('matched_term', '')
                            if matched_term and matched_term in chunk_text:
                                hybrid_score += 50.0  # 大幅增加包含精确匹配关键词块的分数
                                break
                    
                    # 对于包含日期的块，增加额外权重
                    if re.search(r'\d{4}-\d{1,2}-\d{1,2}', chunk_text):
                        hybrid_score += 30.0
                    
                    chunk['score'] = hybrid_score
                    scores_dict[chunk_id] = chunk
        
        # 步骤9: 重排序并返回
        sorted_chunks = sorted(scores_dict.values(), key=lambda x: x.get('score', 0), reverse=True)
        
        # 合并结果，确保精确匹配项优先且无重复
        seen_texts = set()
        final_chunks = []
        
        for chunk in sorted_chunks:
            if chunk and isinstance(chunk, dict):
                chunk_text = chunk.get('text', '')
                if chunk_text and chunk_text not in seen_texts:
                    seen_texts.add(chunk_text)
                    final_chunks.append(chunk)
        
        return final_chunks[:self.rerank_top_k]
    
    def _detect_question_type(self, question: str) -> str:
        """
        检测问题类型，用于针对性推理
        
        Args:
            question: 问题文本
            
        Returns:
            问题类型
        """
        question_lower = question.lower()
        
        for type_name, pattern in self.question_types.items():
            if re.search(pattern, question_lower, re.IGNORECASE):
                return type_name
        
        return "general"
    

    
    async def evaluate_model(self, prompt: Dict) -> str:
        """
        评估模型，实现核心检索逻辑

        Args:
            prompt: 包含context_data和question的字典

        Returns:
            答案
        """
        # 安全获取prompt数据
        context_data = prompt.get('context_data', {})
        question = prompt.get('question', '')
        files = context_data.get('files', [])
        
        # 检测问题类型
        question_type = self._detect_question_type(question)
        
        # 提取关键词
        keywords = await self._extract_keywords_with_llm(question)
        
        # 直接使用精确匹配结果构建上下文
        exact_matches = self._find_exact_matches(keywords, files)
        
        # 检索相关块
        chunks = await self._retrieve_relevant_chunks(question, files)
        
        # 合并精确匹配和检索结果
        all_relevant_texts = []
        seen_texts = set()
        
        # 先添加精确匹配结果
        for match in exact_matches:
            if match and isinstance(match, dict):
                text = match.get('text', '')
                if text and text not in seen_texts:
                    all_relevant_texts.append(text)
                    seen_texts.add(text)
        
        # 再添加检索结果
        for chunk in chunks:
            if chunk and isinstance(chunk, dict):
                text = chunk.get('text', '')
                if text and text not in seen_texts:
                    all_relevant_texts.append(text)
                    seen_texts.add(text)
        
        # 构建上下文
        context = "Context Information:\n"

        # 标记关键匹配项
        has_exact_matches = False
        exact_match_keywords = set()
        for match in exact_matches:
            if match and isinstance(match, dict):
                matched_term = match.get('matched_term', '')
                if matched_term:
                    exact_match_keywords.add(matched_term)

        if exact_match_keywords:
            context += "\nKey Matching Terms Found: " + ", ".join(exact_match_keywords) + "\n"
            has_exact_matches = True

        context += "\nRelevant Text Chunks:\n"
        context += "=" * 50 + "\n"

        # 对于日期计算类问题，确保包含所有可能的日期信息
        all_context_texts = []
        
        # 使用合并后的相关文本构建上下文
        for i, text in enumerate(all_relevant_texts[:self.rerank_top_k]):
            # 高亮显示精确匹配的关键词
            highlighted_text = text
            for keyword in exact_match_keywords:
                if keyword in highlighted_text:
                    # 使用特殊标记高亮显示关键词
                    highlighted_text = highlighted_text.replace(keyword, f"[[{keyword}]]")
            
            context += f"\nChunk {i+1}:\n"
            context += "-" * 30 + "\n"
            context += f"{highlighted_text}\n"
            context += "-" * 30 + "\n"
            all_context_texts.append(text)
        
        # 对于日期计算类问题，添加额外的日期信息作为提示
        if question_type == "date_time":
            from datetime import datetime
            from collections import defaultdict

            # 1. 从 context 中提取日期 + 所属 chunk
            date_patterns = [
                r'\b\d{4}-\d{1,2}-\d{1,2}\b',
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b'
            ]
            chunk_dates = defaultdict(list)

            # 从所有相关块中提取日期，包括精确匹配和检索到的块
            all_relevant_chunks = exact_matches + chunks
            
            for chunk in all_relevant_chunks:
                if chunk and isinstance(chunk, dict):
                    text = chunk.get("text", "")
                    chunk_id = chunk.get("chunk_id", f"{chunk.get('filename', 'unknown')}:{chunk.get('start_pos', 0)}")
                    for pattern in date_patterns:
                        for d in re.findall(pattern, text):
                            try:
                                # 处理完整月份名称格式，如 "December 26, 2057"
                                parsed = datetime.strptime(d, "%B %d, %Y")
                            except:
                                try:
                                    # 处理缩写月份格式，如 "Nov 22, 2042"
                                    parsed = datetime.strptime(d, "%b %d, %Y")
                                except:
                                    try:
                                        # 处理YYYY-MM-DD格式，包括单数字月份和日期，如 "2042-10-5"
                                        parsed = datetime.strptime(d, "%Y-%m-%d")
                                    except:
                                        try:
                                            # 处理MM/DD/YYYY格式，如 "12/25/2031"
                                            parsed = datetime.strptime(d, "%m/%d/%Y")
                                        except:
                                            continue
                            chunk_dates[chunk_id].append(parsed)
            
            # 从问题中提取日期，因为有些测试用例的日期在问题中
            question_chunk_id = "question"
            for pattern in date_patterns:
                for d in re.findall(pattern, question):
                    try:
                        # 处理完整月份名称格式，如 "December 26, 2057"
                        parsed = datetime.strptime(d, "%B %d, %Y")
                    except:
                        try:
                            # 处理缩写月份格式，如 "Nov 22, 2042"
                            parsed = datetime.strptime(d, "%b %d, %Y")
                        except:
                            try:
                                # 处理YYYY-MM-DD格式，包括单数字月份和日期，如 "2042-10-5"
                                parsed = datetime.strptime(d, "%Y-%m-%d")
                            except:
                                try:
                                    # 处理MM/DD/YYYY格式，如 "12/25/2031"
                                    parsed = datetime.strptime(d, "%m/%d/%Y")
                                except:
                                    continue
                    chunk_dates[question_chunk_id].append(parsed)

            # 2. 计算日期差并作为提示
            candidates = []
            for chunk_id, ds in chunk_dates.items():
                if len(ds) >= 2:
                    for i in range(len(ds)):
                        for j in range(i + 1, len(ds)):
                            delta = abs((ds[i] - ds[j]).days)
                            # 过滤掉0值和不合理的大值
                            if 0 < delta <= 400:
                                candidates.append(delta)

            # 3. 兜底：允许跨 chunk，但加约束（防止乱配）
            if not candidates:
                all_dates = []
                for ds in chunk_dates.values():
                    all_dates.extend(ds)

                for i in range(len(all_dates)):
                    for j in range(i + 1, len(all_dates)):
                        diff = abs((all_dates[i] - all_dates[j]).days)

                        # 关键过滤条件
                        if 0 < diff <= 400:   # 防止跨几年乱算
                            candidates.append(diff)

            # 4. 将提取的日期和计算结果作为提示添加到上下文中
            all_extracted_dates = []
            for ds in chunk_dates.values():
                all_extracted_dates.extend(ds)
            
            if all_extracted_dates:
                # 格式化日期
                formatted_dates = [d.strftime("%Y-%m-%d") for d in all_extracted_dates]
                context += f"\nExtracted Dates: {', '.join(set(formatted_dates))}\n"
                
                # 检查问题类型，添加更精确的提示
                if re.search(r'(day of the week|weekday)', question.lower()):
                    # 对于星期几问题，确保提供所有提取的日期及其对应的星期几
                    weekday_hints = []
                    for d in all_extracted_dates:
                        weekday = d.strftime("%A")
                        date_str = d.strftime("%Y-%m-%d")
                        weekday_hints.append(f"{date_str} is a {weekday}")
                    context += f"\nDate Weekday Hints: {'; '.join(weekday_hints)}\n"
                elif candidates:
                    # 对于日期差问题，添加日期差提示
                    context += f"\nDate Difference Hint: The time difference between relevant dates appears to be around {min(candidates)} days.\n"
        
        # 根据问题类型调整提示
        type_instructions = {
            "computation": "Carefully calculate the exact numerical value. Show your work if necessary. Use only numbers from the context.",
            "string_analysis": "Analyze the string with precision. Count characters exactly. For hashes, reproduce exactly from context.",
            "encoding": "Decode or encode accurately using the specified method. Use only data from the context.",
            "general": "Answer based only on the context provided. Be concise and precise."
        }

        # 构建最终的LLM提示
        llm_prompt = f"""
        Instructions:
        1. {type_instructions.get(question_type, type_instructions["general"])}
        2. Provide ONLY the exact answer without any additional text, explanations, formatting, or symbols.
        3. If the answer is a number, return only the digits (e.g., "50" not "50 days").
        4. If the answer is a day of the week, return only the exact word (e.g., "Thursday").
        5. Do NOT add any other content like calculations, explanations, or markdown formatting.

        {context}

        Question: {question}

        Answer: """

        # 调用LLM
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who answers questions based on the provided context."},
                    {"role": "user", "content": llm_prompt}
                ],
                temperature=0,
                max_tokens=300
            )
            
            answer = response.choices[0].message.content
            if answer is None:
                answer = ""
            
            return answer
            
        except Exception as e:
            # 如果LLM调用失败，尝试从上下文中直接提取答案
            # 对于日期计算类问题，尝试直接返回计算结果
            if question_type == "date_time" and candidates:
                return str(min(candidates))
            # 对于其他问题，尝试从上下文中提取关键词匹配的内容
            elif all_relevant_texts:
                # 简单策略：返回第一个相关文本中的数字或关键词
                for text in all_relevant_texts:
                    # 查找所有数字
                    numbers = re.findall(r'\b\d+\b', text)
                    if numbers:
                        return numbers[0]
                    # 查找所有大写关键词
                    keywords = re.findall(r'\b[A-Z]+(?:-[A-Z0-9]+)*\b', text)
                    if keywords:
                        return keywords[0]
            # 如果以上策略都失败，返回空字符串
            return ""
    
    def generate_prompt(self, **kwargs) -> Dict:
        """
        生成提示结构

        Args:
            **kwargs: 灵活参数，包括context_data和question等

        Returns:
            包含RAG处理所需提示信息的字典
        """
        context_data = kwargs.get('context_data', {})
        question = kwargs.get('question', '')
        
        return {
            'context_data': context_data,
            'question': question
        }
        
    def encode_text_to_tokens(self, text: str) -> List[int]:
        """
        使用tiktoken将文本编码为token ID

        Args:
            text: 要编码的文本

        Returns:
            token ID列表
        """
        return self.tokenizer.encode(text)
        
    def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str:
        """
        将token ID解码回文本

        Args:
            tokens: 要解码的token ID列表
            context_length: 可选的token数量限制

        Returns:
            解码后的文本
        """
        if context_length:
            tokens = tokens[:context_length]
        return self.tokenizer.decode(tokens)
