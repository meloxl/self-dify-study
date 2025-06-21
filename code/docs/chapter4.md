### 4.搭建工作流

Agent 工作流（or chatflow）

我们现在可以来做一个小红书读书卡片生成的chatflow。

一、首先创建chatflow，名字命名为小红书读书卡片

二、配置llm，配置两个llm模型平台，配置智谱AI和配置硅基流动API

三、配置Artifacts插件，html直接渲染，生成url地址：http://localhost/e/recwlp7g5fkq9ejn/

四、只是生成了一段页面，没有美化
![chatflow工作流程图](images/chatflow.png)


工作流程如下:
1. 用户输入: 接收用户输入的书籍信息和要点
2. 生成读书笔记: 使用LLM模型生成读书笔记内容
3. 生成HTML: 将笔记内容转换为HTML格式
4. 渲染页面: 使用Artifacts插件渲染HTML,生成可访问的URL
5. 返回结果: 将生成的URL返回给用户


备注：没有完成像文档上一样美化的页面，后续需要看看如何完成。

