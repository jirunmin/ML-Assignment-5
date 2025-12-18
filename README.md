# 作业五：Needle in a Haystack - 多文档信息检索

**环境要求**：Python 3.8+
**评分方式**：主观评分（参考排行榜表现）

**📅 截止日期：11月30日** | **🏆 [查看实时排行榜](http://101.132.193.95:3000/)**

---

## 一、任务概述

### 什么是 Needle in a Haystack？

Needle in a Haystack（大海捞针）是一种评估大语言模型（LLM）长文本理解能力的测试方法。在本次作业中，我们将多个关键信息（needles）随机插入到不同的文本文件（haystack）中，然后测试你实现的 Agent 能否准确检索并回答相关问题。

### 任务目标

实现一个智能 Agent，能够：

- 从多个文本文件中检索相关信息
- 整合不同文件中的信息片段
- 准确回答基于这些信息的问题

---

## 二、系统架构

### 核心组件

**ModelProvider (model.py)**
抽象基类，定义 Agent 必须实现的接口

**LLMMultiNeedleHaystackTester**
多文档测试框架，负责加载文件、插入 needles、调用 Agent、评估答案

**Evaluator**
评分器，支持两种模式：

- `StringMatchEvaluator`：精确字符串匹配（0或1分）
- `LLMEvaluator`：LLM 语义评分（0-10分）

### 测试流程

1. 从 `PaulGrahamEssays` 目录加载所有 `.txt` 文件
2. 随机选择 N 个文件（N = needles 数量）
3. 在每个文件的随机位置插入一个 needle
4. 调用你的 Agent 的 `generate_prompt()` 和 `evaluate_model()`
5. 评估器对答案评分
6. 记录结果和耗时

---

## 三、实现要求
`agents/agent_template.py`实现了一个参考
### 需要实现的接口

继承 `ModelProvider` 并实现以下方法：

```python
async def evaluate_model(self, prompt: Dict) -> str
```

核心方法，接收 prompt 字典，返回答案字符串。

- `prompt['context_data']`：所有文件信息
- `prompt['question']`：需要回答的问题

```python
def generate_prompt(self, **kwargs) -> Dict
```

生成传递给 `evaluate_model()` 的 prompt 结构。

```python
def encode_text_to_tokens(self, text: str) -> List[int]
```

将文本编码为 token IDs（用于计算文本长度）。

```python
def decode_tokens(self, tokens: List[int], context_length: Optional[int] = None) -> str
```

将 token IDs 解码回文本。

---

## 四、环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
API_KEY=你的API密钥
BASE_URL=https://api.example.com/v1
STUDENT_ID=你的学号
STUDENT_NAME=你的姓名
STUDENT_NICKNAME=昵称
MAIN_CONTRIBUTOR=human
```

### 3. 提交评测

```bash
python run.py --agent your_agent:YourAgentClass
```

---

## 五、评分说明

### 评分维度（主观评分）

虽然参考排行榜自动评分，但最终成绩综合考虑：

- **准确率**，**实现质量**，**创新性**，**效率**，**诚信**

### 排行榜

提交后会显示你的平均分数，但排行榜仅供参考，不完全决定最终成绩。

### 🎤 排行榜前三名特别要求

**排行榜前十名的同学可能会被邀请做课堂分享（不强制）：**

1. **课堂分享**：准备 5-10 分钟的方案讲解，内容包括：
   - 你的检索策略和算法设计
   - 关键技术点和创新之处
   - 遇到的挑战和解决方案
   - 性能优化经验

2. **代码提交**：可能会被要求提交完整代码进行审查

---

*如有问题，请及时联系助教。*
*Made with ❤️ for Machine Learning Education*
