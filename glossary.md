# AI Glossary

## Field of AI / Major Types

**Artificial General Intelligence (AGI):** The hypothetical intelligence of a machine that has the capacity to understand or learn any intellectual task that a human being can.

**Artificial Intelligence (AI):** A field of computer science aimed at creating machines capable of performing tasks that typically require human intelligence.

**Computer Vision:** A field of AI that enables computers to interpret and process visual information from the world, similar to human vision.

**Conversational AI:** A subset of AI focused on generating human-like interactions through chatbots and virtual assistants. Utilizing natural language processing, these applications provide responsive communication across platforms like messaging apps and websites, streamlining user engagement.

**Deep Learning (DL):** A branch of machine learning that involves networks capable of learning from data in deep structures, primarily through neural networks with many layers.

**Generative AI (genAI)\:** A branch of AI focused on creating models that can generate new and original content, such as images, music, or text, based on patterns and examples from existing data.

**Large-Language Model (LLM):** An extensive machine learning model trained on vast datasets of text. It excels in understanding, generating, and translating human language by recognizing patterns and nuances.

**Machine Learning (ML):** A subset of AI that focuses on developing algorithms which allow computers to learn from and make predictions or decisions based on data.

**Natural Language Processing (NLP):** A field of AI that gives computers the ability to understand, interpret, and generate human language.

**Recommender Systems:** A subclass of information filtering systems that seek to predict the "rating" or "preference" a user would give to an item.

**Robotics:** The branch involving the design, construction, operation, and application of robots, integrating systems for control, sensory feedback, and information processing.

## Neural Networks & Models

**Attention:** A mechanism that allows a model to weigh the importance of different parts of the input data differently, enabling focused processing on relevant parts for the task at hand.

**Autoencoder**: A type of neural network used to learn efficient data codings in an unsupervised manner.

**Convolutional Neural Networks (CNNs):** A class of deep neural networks, most commonly applied to analyzing visual imagery, characterized by their use of convolutional layers.

**Deep Neural Network:** Machine learning techniques that utilize neural networks with many layers.

**Diffusion Model:** A type of generative model that transforms random noise into structured data, typically images, through a gradual process of learning and refinement.

**Energy-Based Model (EBM):** A framework in machine learning where the system assigns a scalar energy value to each state or configuration of the variables being modeled. The goal is to learn a function that gives lower energy to correct more desirable configurations and higher energy to incorrect or undesirable ones, thereby framing the problem of inference and learning as one of energy minimization.

**Feedforward Neural Networks:** The simplest type of artificial neural network, wherein connections between the nodes do not form a cycle. This model moves information forward from input to output.

**Foundational model:** A baseline model used for a solution set, typically pretrained on large amounts of data using self-supervised learning.

**Generative Adversarial Networks (GANs):** A class of machine learning frameworks designed by pitting two neural networks against each other in a game, typically for the purpose of generating new, synthetic instances of data that can pass for real data.

**Hidden Layer**: Layers of artificial neurons in a neural network that are not directly connected to the input or output.

**Hyperparameters**: These are adjustable model parameters that are tuned in order to obtain optimal performance of the model.

**Latent space:** A representation of compressed data where similar data points are closer together, often used in generative models.

 **Layers:** Hierarchical structures of nodes or neurons that process and transform data sequentially from the input layer to the output, based on weights and activation functions.

**Mixture of Experts (MoE)**: A machine learning technique where several specialized submodels (the “experts”) are trained, and their predictions are combined in a way that depends on the input.

**Model:** A simplified representation of a system or phenomenon, often used in scientific and mathematical contexts to predict behavior or outcomes.

**Model garden:** A collection of pre-built models and algorithms available for adaptation in various tasks, serving as a resource for rapid deployment and experimentation.

**Multimodal**: Pertaining to or involving multiple modes of communication or sensory input.

**Neural Network:** Computing systems inspired by the biological neural networks of animal brains, designed to recognize patterns and solve problems.

**Neuron:** A neuron is a fundamental computational unit in neural networks that processes inputs using a weighted sum and a non-linear activation function to produce an output.

**Node:** A node is a computational point within a neural network that processes inputs and forwards the output to subsequent layers.

**One Shot:** A learning approach where a model is trained on a single example or instance to make predictions or decisions.

**Open Weights:** The practice of sharing the weights of pre-trained models with the public or research community to facilitate research, transparency, and application development without the need for extensive computational resources.

**Parameters**: The adjustable elements in a model that the learning algorithm optimizes to fit the data.

**Pre-trained model:** A model that has been previously trained on a large dataset to solve a problem similar to the one it will be applied to. These models can be fine-tuned with additional data for specific tasks, reducing the need for extensive computational resources and training time.

**Recurrent Neural Networks (RNNs):** A type of neural network where connections between nodes form a directed graph along a temporal sequence, allowing it to exhibit temporal dynamic behavior.

**Self-attention:** A specific form of attention mechanism that enables each part of the input data to interact with and assess the importance of other parts of the same data, thereby improving the model's ability to understand and represent complex relationships.

**Small-Language Model:** A streamlined model trained to handle language tasks efficiently with fewer resources. While less comprehensive than larger models, it remains effective for basic natural language processing applications.

**Vision Transformers (ViT)**: A type of neural network that applies the transformer architecture to image classification tasks.

**Weights**: The coefficients in a neural network that are learned from training data and determine the importance of input features.

## Training/Learning

**Backpropagation:** A method used in training neural networks, where the error is calculated and distributed back through the network to adjust the weights.

**Classification:** A type of supervised learning where the goal is to predict the category or class of an object or event.

**Corpus:** a large and structured set of texts or dataset used for training and evaluating natural language processing models, providing the essential data for learning patterns, structures, and nuances of language.

**Federated Learning:** A machine learning approach where the model is trained across multiple decentralized devices or servers holding local data samples, without exchanging them.

**Fine Tuning**: The process of adjusting a pre-trained model to improve its performance on a specific task.

**Gradient descent**: A cornerstone optimization technique in artificial intelligence (AI), particularly in training machine learning and deep learning models. It iteratively updates model parameters to minimize a loss function, reflecting the model's error on training data, thereby improving the model's accuracy.

**Long Short-Term Memory (LSTM):** A special kind of RNN, capable of learning long-term dependencies, designed to avoid the long-term dependency problem.

**Loss function**: A function that measures the difference between the model's prediction and the actual data, guiding model training.

**Many Shot:** A learning strategy where a model is trained with a large number of examples to improve its accuracy and generalization capabilities.

**Overfitting:** A modeling error in machine learning where a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data.

**Quantum Machine Learning:** Research area that combines quantum computing with machine learning algorithms to process information in fundamentally new ways, aiming to solve complex computational problems more efficiently than classical computers.

**Regularization**: In machine learning, regularization is a technique used to prevent overfitting by adding a penalty term to the model’s loss function. This penalty discourages the model from excessively relying on complex patterns in the training data, promoting more generalizable and less prone-to-overfitting models.

**RLHF (Reinforcement Learning from Human Feedback):** A method to train an AI model by learning from feedback given by humans on model outputs.

**Reinforcement Learning:** An area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize some notion of cumulative reward.

**Reward function**: In reinforcement learning, it quantifies the goal achievement, guiding agents towards desirable actions.

**Self Play:** A method where a model learns by playing against itself, commonly used in game-based AI research.

**Signal:** Any kind of measurable data or input, such as text, images, sound, or numerical values, that algorithms analyze to detect patterns, trends, or insights for making predictions or decisions.

**Supervised Learning:** A type of machine learning where the model is trained on a labeled dataset, which means the algorithm learns from an input-output pair.

**Testing Data:** A subset of data not used to train the model but to test the model's predictions, thereby providing an unbiased evaluation of the model.

**Training Data:** The dataset used to train the model, where the model learns to identify patterns or make decisions.

**Transformer Models:** A type of model introduced in "Attention is All You Need", primarily used in natural language processing, that relies entirely on self-attention mechanisms to draw global dependencies between input and output.

**Transfer Learning:** A machine learning method where a model developed for a task is reused as the starting point for a model on a second task, leveraging pre-learned patterns for better performance or efficiency.

**Underfitting:** Occurs when a model is too simple, both in terms of the algorithm used and the features considered, and fails to capture the underlying trend of the data.

**Unsupervised Learning:** A type of machine learning that deals with input data without labeled responses, aiming to find hidden patterns or intrinsic structures in input data.

**Zero Shot / In Context Learning:** A method where a model makes predictions or understands tasks it has never explicitly been trained on, using general knowledge or context.

## Agents

**Agent:** An entity or actor that takes actions or makes decisions in an environment, often used in computing, economics, and social sciences.

**Agentic:** Pertaining to an agent's ability to act independently, make choices, and impose those choices on the world.

**Reflection**: A process where AI agents evaluate and modify their decision-making strategies based on the outcomes of past actions to enhance future performance.

**System Agent**: An AI entity designed to autonomously perform tasks, interact with digital environments, and adapt to changes to achieve specific objectives efficiently.

**User Proxy Agent**: An AI intermediary that acts on a user's behalf, learning from their preferences and actions to perform tasks and make decisions within digital environments.

## Software/Projects

**AlexNet:** A deep neural network architecture that significantly improved image classification tasks, marking a breakthrough in the field of computer vision.

**AlphaGo:** An AI program developed by DeepMind that plays the board game Go, known for defeating world champion human players.

***Attention is all you need:*** A landmark paper introducing the Transformer model, emphasizing the effectiveness of attention mechanisms without relying on recurrence or convolution.

**CLIP:** (Contrastive Language–Image Pre-training) by OpenAI, is an AI model that understands images in context with textual descriptions, enhancing the ability to generalize across visual concepts.

**CUDA (Compute Unified Device Architecture):** A parallel computing platform and application programming interface (API) model created by NVIDIA, allowing software developers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing.

**ImageNet:** A large visual database used for image classification, recognition, and benchmarking deep neural network models, notably in challenges like the one where AlexNet demonstrated its breakthrough performance.

**Joint Embedding Predictive Architecture (JEPA):** A machine learning framework where models learn to predict the embeddings of one part of an input based on another, using self-supervised learning from unlabeled data​.

**Keras:** An open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library.

**LangChain:** A development framework for Python and JavaScript that simplifies building applications with large language models by providing tools and APIs for tasks like chatbots, context-aware responses, and document summarization.

***Let's Verify Step by Step***: A paper that shows teaching language models to use detailed step feedback is better for solving complex problems than just using final results.

**LORA**: Low Rank Adaptation. A technique to adaptively adjust the learning rates of parameters in optimization algorithms, improving convergence.

**OpenAI GPT (Generative Pre-trained Transformer):** A series of AI models developed by OpenAI that specialize in understanding and generating natural language texts.

**PyTorch:** An open-source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, primarily developed by Facebook's AI Research lab.

**TensorFlow:** An end-to-end open-source platform for machine learning designed by Google. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources that allows researchers to push the state-of-the-art in ML, and developers to easily build and deploy ML-powered applications.

**TPU (Tensor Processing Unit)**: A type of microprocessor developed by Google specifically for accelerating machine learning workloads.

## Misc

**Alignment:** Alignment in AI refers to ensuring AI systems' actions and outputs align with human values and intentions to promote welfare and avoid harm.

**Chain of Thought (CoT)**: A technique where AI generates a sequence of reasoning steps to transition logically from one concept to a conclusion, enhancing transparency and understanding.

**Chat Completions:** Chat completions are the responses generated by conversational AI models to continue or complete a conversation based on user prompts.

**Chatbot:** A software application that simulates human conversation using natural language processing to interact with users.

**Context Window:** The range or extent of surrounding information considered by an algorithm or model to understand or process a specific piece of data.

**Dataset**: A collection of data often structured in a tabular format, comprising variables (columns) and observations (rows), used for analysis, training models, or testing in the field of artificial intelligence and machine learning.

**Embedding**: The representation of data, like words or features, in a high-dimensional space, enabling similarity measurements.

**Emergent Behavior:** Complex behavior or patterns that arise from simpler interactions within a system, not predictable from the individual components alone.

**Human-in-the-Loop (HITL):** A configuration where human feedback is integrated into the AI system’s learning cycle, improving the model’s accuracy and reliability through human expertise.

**Inference:** In genAI, inference is the process of using a trained model to make predictions, serving as the generation step.

**Meta prompting:** An advanced technique involving the generation or manipulation of prompts to improve performance or adaptability of models.

**Moravec's paradox**: The observation that high-level reasoning requires very little computation, but low-level sensorimotor skills require enormous computational resources.

**Planning:** Leveraging the capabilities of LLMs to generate and execute multi-step plans for solving complex tasks.

**Prompt:** The initial context or instruction that sets the task or query for the model.

**Retrieval-Augmented Generation (RAG):** An AI technique that combines a neural network with a retrieval system to enhance language model responses by incorporating relevant information fetched from a large database or corpus.

**Semantic analysis**: The process of understanding the meaning and interpretation of words and sentences in context.

**Singularity:** In the context of AI, the singularity (also known as the technological singularity) refers to a hypothetical future point in time when technological growth becomes uncontrollable and irreversible, leading to unforeseeable changes to human civilization.

**Synthetic Data:** Artificially generated data that mimics the statistical properties of real-world data, used to train machine learning models where actual data may be limited, sensitive, or biased.

**Tensor:** A tensor is a multi-dimensional array used in AI and machine learning to represent complex data structures for neural network computations.

**Token**: The smallest unit of data in NLP, representing words, characters, or subwords.

**Tokenizer**: A tool that splits text into tokens, often used in text processing.

**Variational auto encoder**: A type of autoencoder that generates new data points by learning the distribution of input data.

**Virtual Assistants:** Artificial intelligence systems that can perform tasks or services for an individual based on commands or questions.

## Related Computer Science & Math

**Dot product:** A mathematical operation multiplying corresponding entries of two vectors and summing those products, crucial for measuring similarity or direction differences.

**Edge Computing:** Distributed computing paradigm that brings computation and data storage closer to the location where it is needed, to improve response times and save bandwidth.

**High dimensional space:** Refers to spaces with more than three dimensions, often used to describe datasets with many attributes in complex problem solving.

**Matrix:** A rectangular array of numbers, symbols, or expressions, essential for representing data, performing linear transformations, and facilitating operations in algorithms.

**Physics-Based Computing:** Utilizes physical phenomena to perform computations, often focusing on leveraging the natural properties of physical systems for processing information.

**Thermodynamic Computing:** A theoretical computing paradigm that explores the use of thermodynamic processes for computation, aiming to harness heat and energy flows for information processing.

**Vector:** An object with magnitude and direction, extensively used to represent data points, features, or physical quantities in models and algorithms.

## Words To Add

**Temperature:** A parameter that controls the randomness of model predictions, influencing the diversity and predictability of outputs in generative models.

**Other LLM Inference Settings:** Configuration options used during the inference phase of Large Language Models to tailor the generation process, such as output length and sampling strategies.

**Model IO:** The formats and methods for inputting data into and outputting data from an AI model, including data preprocessing and normalization.

**Softmax:** A function that converts a vector of values into a vector of probabilities, often used in the output layer of neural networks to classify inputs into different categories.

**ONNX (Open Neural Network Exchange):** An open standard for representing machine learning models, allowing models to be transferred between different frameworks and tools.

**LAION:** A dataset and community project aimed at enabling large-scale training of AI models, particularly in the field of generative models, by providing accessible and diverse data.

**Classical Computing:** The traditional model of computing based on binary operations, distinct from quantum computing, forming the foundation for most current AI research and applications.

**VAE (Variational Autoencoder):** A type of autoencoder used for generative tasks, leveraging probabilistic graphical models to encode inputs into a latent space and decode from that space to reconstruct inputs or generate new data.

**Tree of Thought:** An AI research concept aiming to improve the understanding and generation capabilities of models through structured reasoning and hierarchical information processing.

**Bootstrapping Data:** A technique for improving model performance by iteratively refining the training dataset, often using the model's own predictions to generate new training examples.

**AI Engineer:** A professional who specializes in designing, building, and implementing AI models and systems, often involving tasks such as data preprocessing, model training, and deployment.

**Model Distillation:** A process of transferring knowledge from a large, complex model to a smaller, more efficient model, preserving performance while reducing computational demands.

**Epoch:** One complete pass through the entire training dataset by a machine learning algorithm, used as a measure of the extent of training.

**Checkpoint:** A saved state of a model during training, allowing recovery or resumption of training from that point, and facilitating model evaluation at various stages of training.

**Quantization:** The process of reducing the precision of the weights and activations of a model to lower bit widths, which decreases model size and speeds up inference, with minimal impact on accuracy.

**Batch:** A subset of the training dataset that is used in one iteration of model training, allowing for efficient and parallelizable processing.

**Learning Rate:** A hyperparameter that determines the step size at each iteration while moving toward a minimum of the loss function, influencing how quickly a model learns.

**Regularization:** Techniques used to prevent overfitting by imposing constraints on the quantity and type of information your model can store.

**Validation Set:** A subset of the dataset, separate from the training set, used to evaluate the model's performance and tune hyperparameters without using the test set.

**Momentum:** A technique that helps accelerate the gradient descent algorithm by navigating along the relevant direction and dampening oscillations, improving convergence speed.

**Dropout:** A regularization method where randomly selected neurons are ignored during training, preventing them from co-adapting too much and reducing the risk of overfitting.

**Embedding Layer:** A layer in a neural network that transforms large sparse vectors into a lower-dimensional space where similar values are close to each other, often used for processing text or categorical data.
