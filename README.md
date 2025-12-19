# 作业五：Needle in a Haystack - 多文档信息检索

**环境要求**：Python 3.8+  
**评分方式**：主观评分（大模型辅助评分 + 人工审查）  
**📅 截止日期**：12月28日 | **🏆 [查看实时排行榜](http://101.132.193.95:3000/)**

---

## 一、任务概述

### 什么是 Needle in a Haystack？

Needle in a Haystack（大海捞针）是评估大语言模型（LLM）长文本理解能力的测试方法。本次作业将多个关键信息（needles）随机插入不同文本文件（haystack）中，测试你实现的 Agent 能否准确检索并回答相关问题。

### 任务目标

实现一个智能 Agent，能够：
- 从多个文本文件中检索相关信息
- 整合不同文件中的信息片段
- 准确回答基于这些信息的问题

---

## 二、系统架构

### 核心组件

**ModelProvider (`model.py`)**  
抽象基类，定义 Agent 必须实现的接口。

**LLMMultiNeedleHaystackTester**  
多文档测试框架，负责加载文件、插入 needles、调用 Agent、评估答案。

**Evaluator**  
评分器，支持两种模式：
- `StringMatchEvaluator`：精确字符串匹配（0或1分）
- `LLMEvaluator`：LLM 语义评分（0-10分）
默认使用StringMatchEvaluator，一定要看evaluators/string_match_evaluator.py了解评测规则

### 测试流程

1. 从 `PaulGrahamEssays` 目录加载所有 `.txt` 文件
2. 随机选择 N 个文件（N = needles 数量）
3. 在每个文件的随机位置插入一个 needle
4. 调用 Agent 的 `generate_prompt()` 和 `evaluate_model()`
5. 评估器对答案评分
6. 记录结果和耗时

---

## 三、实现要求

参考实现见 `agents/agent_template.py`。

### 必须实现的接口

继承 `ModelProvider` 并实现以下方法：

#### 1. `async def evaluate_model(self, prompt: Dict) -> str`
核心方法，接收 prompt 字典，返回答案字符串。
- `prompt['context_data']`：所有文件信息
- `prompt['question']`：需要回答的问题

#### 2. `def generate_prompt(self, **kwargs) -> Dict`
生成传递给 `evaluate_model()` 的 prompt 结构。

#### 3. `def encode_text_to_tokens(self, text: str) -> List[int]`
将文本编码为 token IDs（用于计算文本长度）。

#### 4. `def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str`
将 token IDs 解码回文本。

---

## 四、⚠️ 诚信规则

### 禁止行为（违规成绩为0分）

1. **禁止读取原始测试文件**
   - 不得读取 `PaulGrahamEssays` 目录中的原始 `.txt` 文件
   - 不得通过比对原始文件和修改后文本来定位 needles

2. **禁止读取测试用例**
   - 不得读取 `test_cases/test_cases_all_en.json` 或其他测试用例文件
   - 不得提前获知测试问题和答案

3. **禁止其他作弊行为**

### 合法使用

以下做法是**允许**的：
- 使用开源工具和库（LangChain、LlamaIndex、向量数据库等）
- 使用 AI 工具辅助编程（GitHub Copilot、ChatGPT 等）
- 参考公开的技术文档和教程
- 与同学讨论技术思路（但不能直接共享代码）

---

## 五、环境配置与提交

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥和个人信息

在项目根目录创建 `.env` 文件：

```env
API_KEY=你的API密钥
BASE_URL=https://chat.ecnu.edu.cn/open/api/v1
STUDENT_ID=你的学号
STUDENT_NAME=你的姓名
STUDENT_NICKNAME=昵称（将显示在排行榜上）
MAIN_CONTRIBUTOR=human
```

### 3. 提交评测

```bash
python submit.py --agent your_agent:YourAgentClass
```

---

## 六、评分说明

### 评分方式

采用**大模型辅助评分 + 人工审查**：

1. **自动化评分**：排行榜显示测试平均分
2. **AI 辅助评分**：大模型分析代码质量、创新性等
3. **人工审查**：教师最终审查并确定成绩

### 评分维度（满分20分）

**前置条件：诚信为本，违规直接0分**

- **准确率**（25%）
- **实现质量**（50%）
- **创新性**（25%）

**最终解释权归机器学习课程团队所有**

### AI 评分标准

大模型将从以下维度评估代码：

**1. 实现质量**
- 是否完成作业要求

**2. 创新性**
- 是否使用独特的检索策略
- 算法设计是否有亮点
- 是否有效利用工具和技术
- 解决方案的创造性

**不诚信行为判定（成绩为0分）：**
- 读取 `PaulGrahamEssays/*.txt` 原始文件
- 通过文件差异比对（diff）定位 needles
- 读取测试用例文件
- 硬编码测试问题或答案
- 使用外部服务预先获取答案
- 规避评测系统检查机制
- 伪造测试结果
- 其他作弊行为

### 排行榜说明

- 排行榜显示测试平均分（准确率部分）
- 排行榜**仅供参考**，不完全决定最终成绩
- 最终成绩 = 准确率 + AI评分 + 人工审查

### 🎤 排行榜前十名特别要求

前十名同学可能被邀请做课堂分享（不强制）：

**1. 课堂分享（5-10分钟）**
- 检索策略和算法设计
- 关键技术点和创新之处
- 遇到的挑战和解决方案
- 性能优化经验

**2. 代码审查**
- 需提交完整代码并接受详细审查

---

*如有问题，请及时联系助教。*  
*Made with ❤️ for Machine Learning Education*
