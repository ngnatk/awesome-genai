# AWSome GenAI

### Description

This is intended to be a community-curated list of GenAI demos & code samples located in the following Github repositories

- [https://github.com/aws/](https://github.com/aws/)
- [https://github.com/awslabs/](https://github.com/awslabs/)
- [https://github.com/aws-samples/](https://github.com/aws-samples/)
- [https://github.com/aws-solutions/](https://github.com/aws-solutions/)
- [https://github.com/build-on-aws/](https://github.com/build-on-aws/)


### Documentation

- [Single prompt inference](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-invoke.html)
- [Model inference parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html)
  - [Anthropic Claude Messages API](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html)
  - [Cohere Embed Models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html)
  - [Meta Llama Models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html)
  - [Mistral AI Models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html)
  - [Stability.ai Diffusion 1.0 Text2Image](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-1-0-text-image.html)
  - [Stability.ai Diffusion 1.0 Image2Image](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-1-0-image-image.html)
- [Prompt templates and examples for Amazon Bedrock text models](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-templates-and-examples.html)


### Chatbot Apps

- [AWS GenAI Chatbot](https://aws-samples.github.io/aws-genai-llm-chatbot/) - A modular and comprehensive solution to deploy a Multi-LLM and Multi-RAG powered chatbot (Amazon Bedrock, Anthropic, HuggingFace, OpenAI, Meta, AI21, Cohere) using AWS CDK on AWS. [https://github.com/aws-samples/aws-genai-llm-chatbot](https://github.com/aws-samples/aws-genai-llm-chatbot)
- Bedrock Agent Apointment Manager DynamoDB [https://github.com/build-on-aws/bedrock-agent-appointment-manager-dynamodb](https://github.com/build-on-aws/bedrock-agent-appointment-manager-dynamodb)
- Bedrock Agents Streamlit [https://github.com/build-on-aws/bedrock-agents-streamlit](https://github.com/build-on-aws/bedrock-agents-streamlit)
- Bedrock Claude Chat [https://github.com/aws-samples/bedrock-claude-chat](https://github.com/aws-samples/bedrock-claude-chat)
- Bedrock Claude Chatbot [https://github.com/aws-samples/bedrock-claude-chatbot](https://github.com/aws-samples/bedrock-claude-chatbot)
- Bedrock Claude Coach [https://github.com/aws-samples/bedrock-claude-codecoach](https://github.com/aws-samples/bedrock-claude-codecoach)
- GenAI Playground Common Design Patterns [https://github.com/aws-samples/genai-playground-common-design-patterns](https://github.com/aws-samples/genai-playground-common-design-patterns)
- GenAI Chatbot Using Bedrock Agents PoC [github.com/aws-samples/genai-chatbot-using-bedrock-agents-poc](https://github.com/aws-samples/genai-chatbot-using-bedrock-agents-poc)
- GenAI Quickstart PoCs [https://github.com/aws-samples/genai-quickstart-pocs](https://github.com/aws-samples/genai-quickstart-pocs)
- [Generative AI Application Builder on AWS](https://aws.amazon.com/solutions/implementations/generative-ai-application-builder-on-aws/) facilitates the development, rapid experimentation, and deployment of generative artificial intelligence (AI) applications without requiring deep experience in AI. The solution includes integrations with Amazon Bedrock and its included LLMs, such as Amazon Titan, and pre-built connectors for 3rd-party LLMs. [https://github.com/aws-solutions/generative-ai-application-builder-on-aws](https://github.com/aws-solutions/generative-ai-application-builder-on-aws)
- [Generative AI CDK Constructs](https://awslabs.github.io/generative-ai-cdk-constructs/) - sample implementations of AWS CDK for common generative AI patterns. [https://github.com/awslabs/generative-ai-cdk-constructs/](https://github.com/awslabs/generative-ai-cdk-constructs/). [Youtube](https://www.youtube.com/watch?v=NI1F4Xxqyr8)
- Generative AI CDK Constructs Samples [github.com/aws-samples/generative-ai-cdk-constructs-samples/](https://github.com/aws-samples/generative-ai-cdk-constructs-samples/)
- Generative AI SageMaker CDK Demo [https://github.com/aws-samples/generative-ai-sagemaker-cdk-demo](https://github.com/aws-samples/generative-ai-sagemaker-cdk-demo)
- PACE GenAI Demos [https://github.com/aws-samples/pace-genai-demos](https://github.com/aws-samples/pace-genai-demos)
- Python FM Playground [https://github.com/build-on-aws/python-fm-playground](https://github.com/build-on-aws/python-fm-playground)
- QnABot on AWS [https://github.com/aws-samples/qnabot-on-aws-plugin-samples](https://github.com/aws-samples/qnabot-on-aws-plugin-samples)


### Benchmarking & Evaluation

- [Agent Evaluation](https://awslabs.github.io/agent-evaluation/). [https://github.com/awslabs/agent-evaluation](https://github.com/awslabs/agent-evaluation)
- FM Bench - Benchmark any Foundation Model (FM) on any AWS service [Amazon SageMaker, Amazon Bedrock, Amazon EKS, Bring your own endpoint etc.]. [https://github.com/aws-samples/foundation-model-benchmarking-tool](https://github.com/aws-samples/foundation-model-benchmarking-tool). <a href="https://pypi.org/project/fmbench/" target="_blank"><img src="https://img.shields.io/pypi/v/fmbench.svg" alt="PyPI Version"></a>
- [FM Eval](https://aws.github.io/fmeval/fmeval.html). [https://github.com/aws/fmeval/](https://github.com/aws/fmeval/)
- FM Evaluation at Scale [https://github.com/aws-samples/fm-evaluation-at-scale](https://github.com/aws-samples/fm-evaluation-at-scale)
- [LLM Evaluation Methodology](https://github.com/aws-samples/llm-evaluation-methodology)
- RefChecker [https://github.com/amazon-science/RefChecker](https://github.com/amazon-science/RefChecker)


### Libraries

- **Distill CLI** - the Distill CLI uses Amazon Transcribe and Amazon Bedrock to create summaries of your audio recordings (e.g., meetings, podcasts, etc.) directly from the command line. It is based on the open source tool: [Amazon Bedrock Audio Summarizer](https://github.com/aws-samples/amazon-bedrock-audio-summarizer). [github.com/awslabs/distill-cli](https://github.com/awslabs/distill-cli)
- **Rhubarb** - A Python framework for multi-modal document understanding with Amazon Bedrock [github.com/awslabs/rhubarb](https://github.com/awslabs/rhubarb)


### Solutions & Sample Code

- Advanced RAG Router with Amazon Bedrock - [https://github.com/aws-samples/advanced-rag-router-with-amazon-bedrock](https://github.com/aws-samples/advanced-rag-router-with-amazon-bedrock)
- Amazon Bedrock Audio Summarizer. [github.com/aws-samples/amazon-bedrock-audio-summarizer](https://github.com/aws-samples/amazon-bedrock-audio-summarizer)
- Amazon Bedrock Industry Use Cases - [Healthcare Life Sciences](https://github.com/aws-samples/amazon-bedrock-industry-use-cases/blob/main/healthcare-life-sciences) and [Travel & Hospitality](https://github.com/aws-samples/amazon-bedrock-industry-use-cases/blob/main/travel-hospitality). [https://github.com/aws-samples/amazon-bedrock-industry-use-cases](https://github.com/aws-samples/amazon-bedrock-industry-use-cases)
- Amazon Bedrock Samples [https://github.com/aws-samples/amazon-bedrock-samples](https://github.com/aws-samples/amazon-bedrock-samples)
- Amazon Bedrock Serverless Prompt Chaining [https://github.com/aws-samples/amazon-bedrock-serverless-prompt-chaining](https://github.com/aws-samples/amazon-bedrock-serverless-prompt-chaining)
- Amazon SageMaker Generative AI [https://github.com/aws-samples/amazon-sagemaker-generativeai](https://github.com/aws-samples/amazon-sagemaker-generativeai)
- Amazon Transcribe Post Call Analytics (PCA) [https://github.com/aws-samples/amazon-transcribe-post-call-analytics](https://github.com/aws-samples/amazon-transcribe-post-call-analytics)
- Aurora PostgreSQL pgvector [https://github.com/aws-samples/aurora-postgresql-pgvector/](https://github.com/aws-samples/aurora-postgresql-pgvector/)
- AWS LLM Gateway [https://github.com/aws-samples/llm-gateway](https://github.com/aws-samples/llm-gateway)
- AWS Video Transcriber [https://github.com/awslabs/aws-video-transcriber](https://github.com/awslabs/aws-video-transcriber)
- Bedrock Agent txt2sql [https://github.com/build-on-aws/bedrock-agent-txt2sql](https://github.com/build-on-aws/bedrock-agent-txt2sql)
- Bedrock Knowledgebase Agent Workload IaC [https://github.com/aws-samples/amazon-bedrock-rag-knowledgebases-agents-cloudformation](https://github.com/aws-samples/amazon-bedrock-rag-knowledgebases-agents-cloudformation)
- Bedrock Mistral prompting examples [https://github.com/aws-samples/bedrock-mistral-prompting-examples](https://github.com/aws-samples/bedrock-mistral-prompting-examples)
- Enhanced Document Understanding on AWS [https://github.com/aws-solutions/enhanced-document-understanding-on-aws](https://github.com/aws-solutions/enhanced-document-understanding-on-aws)
- Generative AI Amazon Bedrock Langchain Agent Example [https://github.com/aws-samples/generative-ai-amazon-bedrock-langchain-agent-example](https://github.com/aws-samples/generative-ai-amazon-bedrock-langchain-agent-example)
- Lambda Bedrock Response Streaming [https://github.com/aws-samples/serverless-patterns/tree/main/lambda-bedrock-response-streaming](https://github.com/aws-samples/serverless-patterns/tree/main/lambda-bedrock-response-streaming)
- Langchain Embeddings - Building a Multimodal Search Engine for Text and Image with Amazon Titan Embeddings, Amazon Bedrock, Amazon Aurora and LangChain [github.com/build-on-aws/langchain-embeddings](https://github.com/build-on-aws/langchain-embeddings)
- Media Analysis Policy Evaluation Framework [https://github.com/aws-samples/media-analysis-policy-evaluation-framework](https://github.com/aws-samples/media-analysis-policy-evaluation-framework)
- Simplified Corrective RAG [https://github.com/aws-samples/simplified-corrective-rag](https://github.com/aws-samples/simplified-corrective-rag)
- Video Understanding Solution [https://github.com/aws-samples/video-understanding-solution](https://github.com/aws-samples/video-understanding-solution)


### Workshops

- Amazon Bedrock Agents Quickstart [https://github.com/build-on-aws/amazon-bedrock-agents-quickstart](https://github.com/build-on-aws/amazon-bedrock-agents-quickstart)
- Amazon Bedrock Retrieval-Augmented Generation (RAG) Workshop [https://github.com/aws-samples/amazon-bedrock-rag-workshop](https://github.com/aws-samples/amazon-bedrock-rag-workshop)
- Amazon Bedrock Workshop [https://github.com/aws-samples/amazon-bedrock-workshop](https://github.com/aws-samples/amazon-bedrock-workshop)
- Building Generative AI Apps on AWS with CDK [https://catalog.workshops.aws/building-genai-apps/en-US](https://catalog.workshops.aws/building-genai-apps/en-US)
- Building with Amazon Bedrock and LangChain [https://catalog.us-east-1.prod.workshops.aws/workshops/cdbce152-b193-43df-8099-908ee2d1a6e4/en-US](https://catalog.us-east-1.prod.workshops.aws/workshops/cdbce152-b193-43df-8099-908ee2d1a6e4/en-US)
- Data Science on AWS [https://github.com/aws-samples/data-science-on-aws](https://github.com/aws-samples/data-science-on-aws)
- LLM Ops Workshop - Streamline LLM operations using Amazon SageMaker. [https://github.com/aws-samples/llmops-workshop](https://github.com/aws-samples/llmops-workshop)
- Operationalize Generative AI Applications using LLMOps [https://catalog.us-east-1.prod.workshops.aws/workshops/90992473-01e8-42d6-834f-9baf866a9057/en-US/](https://catalog.us-east-1.prod.workshops.aws/workshops/90992473-01e8-42d6-834f-9baf866a9057/en-US/)
- Prompt Engineering with Anthropic's Claude 3 [https://catalog.workshops.aws/prompt-eng-claude3/en-US](https://catalog.workshops.aws/prompt-eng-claude3/en-US)
- Txt to SQL Bedrock Workshop [https://github.com/aws-samples/text-to-sql-bedrock-workshop/](https://github.com/aws-samples/text-to-sql-bedrock-workshop/)


### Blog Posts with Git Repositories

In reverse chronological order:

- 2024.05.31 [Pre-training genomic language models using AWS HealthOmics and Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/pre-training-genomic-language-models-using-aws-healthomics-and-amazon-sagemaker/). Code: [github.com/aws-samples/genomic-language-model-pretraining-with-healthomics-seq-store](https://github.com/aws-samples/genomic-language-model-pretraining-with-healthomics-seq-store)
- 2024.05.30 [Dynamic video content moderation and policy evaluation using AWS generative AI services](https://aws.amazon.com/blogs/machine-learning/dynamic-video-content-moderation-and-policy-evaluation-using-aws-generative-ai-services/). Code: [github.com/aws-samples/media-analysis-policy-evaluation-framework](https://github.com/aws-samples/media-analysis-policy-evaluation-framework)
- 2024.05.29 [Fine-tune large multimodal models using Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/fine-tune-large-multimodal-models-using-amazon-sagemaker/). Code: [github.com/aws-samples/amazon-sagemaker-finetune-deploy-llava-huggingface](https://github.com/aws-samples/amazon-sagemaker-finetune-deploy-llava-huggingface)
- 2024.05.24 [Building Generative AI prompt chaining workflows with human in the loop](https://aws.amazon.com/blogs/machine-learning/building-generative-ai-prompt-chaining-workflows-with-human-in-the-loop/). Code: [github.com/aws-samples/serverless-genai-examples/tree/main/prompt-chaining-with-stepfunctions](https://github.com/aws-samples/serverless-genai-examples/tree/main/prompt-chaining-with-stepfunctions)
- 2024.05.23 [Accelerate Mixtral 8x7B pre-training with expert parallelism on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/accelerate-mixtral-8x7b-pre-training-with-expert-parallelism-on-amazon-sagemaker/). Jupyter Notebook: [smp-train-mixtral-fsdp-ep.ipynb](https://github.com/aws/amazon-sagemaker-examples/blob/main/training/distributed_training/pytorch/model_parallel_v2/mixtral/smp-train-mixtral-fsdp-ep.ipynb)
- 2024.05.21 [Efficient and cost-effective multi-tenant LoRA serving with Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/efficient-and-cost-effective-multi-tenant-lora-serving-with-amazon-sagemaker/). Jupyter Notebook: [llama2-7b-mistral-7b-multi-lora-adapters.ipynb](https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/LoRA-Adapters-IC/llama2-7b-mistral-7b-multi-lora-adapters.ipynb)
- 2024.05.21 [Create a multimodal assistant with advanced RAG and Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/create-a-multimodal-assistant-with-advanced-rag-and-amazon-bedrock/). Code: [github.com/alfredcs/mmrag](https://github.com/alfredcs/mmrag)
- 2024.05.07 [Boost employee productivity with automated meeting summaries using Amazon Transcribe, Amazon SageMaker, and LLMs from Hugging Face](https://aws.amazon.com/blogs/machine-learning/boost-employee-productivity-with-automated-meeting-summaries-using-amazon-transcribe-amazon-sagemaker-and-llms-from-hugging-face/). Code: [github.com/aws-samples/audio-conversation-summary-with-hugging-face-and-transcribe](https://github.com/aws-samples/audio-conversation-summary-with-hugging-face-and-transcribe)
- 2024.04.19 [Talk to your slide deck using multimodal foundation models hosted on Amazon Bedrock – Part 2](https://aws.amazon.com/blogs/machine-learning/talk-to-your-slide-deck-using-multimodal-foundation-models-hosted-on-amazon-bedrock-and-amazon-sagemaker-part-2/). Code: [github.com/aws-samples/multimodal-rag-on-slide-decks/tree/main/Blog2-LVM-TitanEmbeddings](https://github.com/aws-samples/multimodal-rag-on-slide-decks/tree/main/Blog2-LVM-TitanEmbeddings)
- 2024.04.18 [Live Meeting Assistant with Amazon Transcribe, Amazon Bedrock, and Knowledge Bases for Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/live-meeting-assistant-with-amazon-transcribe-amazon-bedrock-and-knowledge-bases-for-amazon-bedrock/). Code: [github.com/aws-samples/amazon-transcribe-live-meeting-assistant](https://github.com/aws-samples/amazon-transcribe-live-meeting-assistant)
- 2024.03.13 [Moderate audio and text chats using AWS AI services and LLMs](https://aws.amazon.com/blogs/machine-learning/moderate-audio-and-text-chats-using-aws-ai-services-and-llms/). Code: [github.com/aws-samples/aws-genai-audio-text-chat-moderation](https://github.com/aws-samples/aws-genai-audio-text-chat-moderation)
- 2024.02.28 [Build a robust text-to-SQL solution generating complex queries, self-correcting, and querying diverse data sources](https://aws.amazon.com/blogs/machine-learning/build-a-robust-text-to-sql-solution-generating-complex-queries-self-correcting-and-querying-diverse-data-sources/). Code: [github.com/aws-samples/text-to-sql-for-athena](https://github.com/aws-samples/text-to-sql-for-athena)
- 2024.02.19 [Build a contextual chatbot application using Knowledge Bases for Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/build-a-contextual-chatbot-application-using-knowledge-bases-for-amazon-bedrock/). Code: [github.com/aws-samples/amazon-bedrock-samples/tree/main/rag-solutions/contextual-chatbot-using-knowledgebase](https://github.com/aws-samples/amazon-bedrock-samples/tree/main/rag-solutions/contextual-chatbot-using-knowledgebase)
- 2024.01.30 [Talk to your slide deck using multimodal foundation models hosted on Amazon Bedrock and Amazon SageMaker – Part 1](https://aws.amazon.com/blogs/machine-learning/talk-to-your-slide-deck-using-multimodal-foundation-models-hosted-on-amazon-bedrock-and-amazon-sagemaker-part-1/). Code: [github.com/aws-samples/multimodal-rag-on-slide-decks/tree/main/Blog1-TitanEmbeddings-LVM](https://github.com/aws-samples/multimodal-rag-on-slide-decks/tree/main/Blog1-TitanEmbeddings-LVM)
- 2023.12.13 [Create summaries of recordings using generative AI with Amazon Bedrock and Amazon Transcribe](https://aws.amazon.com/blogs/machine-learning/create-summaries-of-recordings-using-generative-ai-with-amazon-bedrock-and-amazon-transcribe/). Code: [github.com/aws-samples/amazon-bedrock-samples/tree/main/generative-ai-solutions/recordings-summary-generator](https://github.com/aws-samples/amazon-bedrock-samples/tree/main/generative-ai-solutions/recordings-summary-generator)
- 2023.11.29 [Amazon SageMaker Clarify makes it easier to evaluate and select foundation models](https://aws.amazon.com/blogs/aws/amazon-sagemaker-clarify-makes-it-easier-to-evaluate-and-select-foundation-models-preview/). Code: [github.com/aws/fmeval/tree/main/examples](https://github.com/aws/fmeval/tree/main/examples)
- 2023.11.29 [Operationalize LLM Evaluation at Scale using Amazon SageMaker Clarify and MLOps services](https://aws.amazon.com/blogs/machine-learning/operationalize-llm-evaluation-at-scale-using-amazon-sagemaker-clarify-and-mlops-services/). Code: [github.com/aws-samples/fm-evaluation-at-scale](https://github.com/aws-samples/fm-evaluation-at-scale)
- 2021.12.17 (updated Jan 2024 v0.7.5) [Post call analytics for your contact center with Amazon language AI services](https://aws.amazon.com/blogs/machine-learning/post-call-analytics-for-your-contact-center-with-amazon-language-ai-services/). Code: [github.com/aws-samples/amazon-transcribe-post-call-analytics](https://github.com/aws-samples/amazon-transcribe-post-call-analytics)
