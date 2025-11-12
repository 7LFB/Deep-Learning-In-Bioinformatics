# Deep-Learning-In-Bioinformatics
# 面向计算机相关专业本科生的生物医学信息学课程

授课教师：Chong 
海南大学 计算机科学与技术学院/软件学院

## 1. 课程简介 （建设中...）

生物医学信息学是一门融合计算机科学、数据科学、人工智能与生物医学的新兴交叉学科，致力于运用先进的计算技术和算法解决生物医学领域的复杂问题。随着高通量测序技术、医学影像技术、可穿戴设备和电子健康档案系统的快速发展，生物医学领域正在产生海量的异构数据，对计算机专业人才提出了迫切需求。本课程响应国家**"新医科"**建设及**"医工交叉融合"**的政策导向，依托于学校在计算机科学、人工智能和大数据技术方面的学科优势，面向计算机相关专业学生开设，旨在培养既懂计算技术又了解生物医学应用场景的复合型人才。

该课程面向计算机科学与技术、软件工程、数据科学、人工智能等专业的本科生，一般在本科大三或大四阶段进行讲授，目前共设48个课时，分16周讲授。课程从计算机专业学生的知识背景出发，重点讲解生物医学领域的数据特征、计算挑战、算法设计和系统开发方法，涵盖序列分析算法、医学图像处理、临床数据挖掘、药物计算设计等多个应用方向，并精选前沿文献和开源项目供学生进行拓展学习。课程目的是帮助计算机专业学生理解生物医学问题的计算本质，掌握领域特定的算法和工具，培养跨学科问题求解能力。该课程要求学生具备扎实的数据结构与算法基础，掌握至少一门编程语言（Python、Java、C++等），了解基本的机器学习和数据库知识。

本课程采用的PPT教案为中英文双语，讲授为双语教学。教学内容参考了CMU、斯坦福大学、MIT等顶尖计算机院系开设的计算生物学和医疗健康信息学课程，同时结合国内生物医学研究和医疗信息化的实际需求进行了优化设计。课程强调"问题驱动、算法为核心、工程实践为导向"的教学理念，每个主题都配有真实数据集、开源工具和编程项目，帮助学生从计算机专业视角理解和解决生物医学问题。

后续将上传本课程的教学视频、代码仓库和项目案例，敬请期待。

---

## 2. 课程教材

### 中文教材

- 《生物信息学算法导论》（原书第2版）[美] Neil C. Jones等著
- 《计算生物学：基因组学、蛋白质组学与药物设计》李霞等译
- 《医疗健康大数据：技术、应用与治理》王金桥等著

### 英文教材

- _Bioinformatics Algorithms: An Active Learning Approach_ 3rd Edition by Compeau & Pevzner
- _Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids_ by Durbin et al.
- _Deep Learning for the Life Sciences_ by Bharath Ramsundar et al.
- _Biomedical Data Science_ by Berman et al.

### 补充算法与工程参考

- _Algorithms on Strings, Trees, and Sequences_ by Dan Gusfield
- _Mining of Massive Datasets_ by Leskovec, Rajaraman & Ullman
- _Designing Data-Intensive Applications_ by Martin Kleppmann
- _Python for Bioinformatics_ 2nd Edition by Sebastian Bassi

---

## 3. 教学大纲

| 课时安排 | 内容 | 课件 | 阅读材料 | 编程项目/作业 |
|---------|------|------|---------|--------------|
| 课时-1 | 课程导论与计算视角<br>- 生物医学信息学的计算挑战<br>- 数据规模与复杂性<br>- 典型应用场景<br>- 职业发展与产业机会 | 课件1 | Nature Biotechnology综述<br>Science计算生物学专题<br>医疗AI独角兽公司案例<br>开源生物信息学项目推荐 | 环境配置：<br>Anaconda/Docker<br>Jupyter Notebook |
| 课时-2 | 生物学基础速成<br>- DNA/RNA/蛋白质基础概念<br>- 中心法则与基因表达<br>- 测序技术原理<br>- 生物医学数据格式(FASTA/FASTQ/SAM/BAM/VCF) | 课件2 | 生物学速成教程<br>NCBI/EBI数据库介绍<br>Biopython文档<br>数据格式标准规范 | 作业1：<br>生物序列文件解析器<br>格式转换工具开发 |
| 课时-3 | 序列比对算法<br>- 动态规划基础<br>- 全局比对(Needleman-Wunsch)<br>- 局部比对(Smith-Waterman)<br>- 启发式算法(BLAST/FASTA)<br>- 多序列比对 | 课件3 | 经典算法论文<br>BLAST算法详解<br>比对算法时间复杂度分析<br>GPU加速比对工具 | 项目1：<br>实现序列比对算法<br>性能优化与并行化 |
| 课时-4 | 字符串算法与索引结构<br>- 后缀树/后缀数组<br>- Burrows-Wheeler变换<br>- FM-Index<br>- 短序列比对(BWA/Bowtie) | 课件4 | 字符串算法经典论文<br>BWA算法原理<br>基因组索引构建<br>压缩数据结构 | 项目2：<br>基于BWT的序列搜索<br>基因组比对工具 |
| 课时-5 | 图算法在生物信息学中的应用<br>- 序列组装问题<br>- De Bruijn图<br>- 欧拉路径与哈密尔顿路径<br>- 基因组组装工具(SPAdes/Velvet) | 课件5 | 基因组组装综述<br>图算法在组装中的应用<br>长读长vs短读长技术<br>宏基因组组装挑战 | 项目3：<br>简化的基因组组装器<br>De Bruijn图构建 |
| 课时-6 | 隐马尔可夫模型(HMM)<br>- HMM基础理论<br>- Forward/Backward算法<br>- Viterbi算法<br>- 基因预测与蛋白质结构域识别 | 课件6 | HMM经典教程<br>HMMER工具文档<br>基因预测算法比较<br>Pfam数据库介绍 | 作业2：<br>HMM实现与应用<br>CpG岛识别 |
| 课时-7 | 机器学习在基因组学中的应用<br>- 特征工程与表示学习<br>- 变异效应预测<br>- 转录因子结合位点预测<br>- 基因表达数据分析 | 课件7 | Nature Methods ML专题<br>DeepSEA/Basset论文<br>scikit-learn生物应用<br>特征选择方法 | 项目4：<br>基因表达分类器<br>变异致病性预测 |
| 课时-8 | 深度学习与序列建模<br>- CNN在序列分析中的应用<br>- RNN/LSTM/GRU<br>- Transformer与注意力机制<br>- 预训练模型(DNA-BERT/ProtBERT) | 课件8 | DeepBind论文<br>Enformer架构解析<br>生物序列预训练模型<br>PyTorch/TensorFlow实践 | 项目5：<br>基于深度学习的<br>启动子识别系统 |
| 课时-9 | 医学图像处理基础<br>- DICOM标准与医学影像格式<br>- 图像预处理与增强<br>- 图像分割算法<br>- 配准与融合技术 | 课件9 | 医学图像处理综述<br>ITK/SimpleITK文档<br>3D Slicer使用教程<br>影像数据集介绍 | 项目6：<br>CT/MRI图像分割<br>器官体积计算 |
| 课时-10 | 深度学习与医学影像<br>- U-Net及其变体<br>- ResNet/DenseNet在医学影像中的应用<br>- 目标检测(YOLO/Faster R-CNN)<br>- 迁移学习与小样本学习 | 课件10 | U-Net论文精读<br>医学影像AI综述<br>数据增强技术<br>模型可解释性方法 | 项目7：<br>肺结节检测系统<br>病理图像分类 |
| 课时-11 | 临床数据挖掘<br>- 电子病历数据处理<br>- 时间序列分析<br>- 生存分析与风险预测<br>- 缺失数据处理 | 课件11 | MIMIC数据库介绍<br>临床预测模型论文<br>医疗NLP技术<br>隐私保护技术 | 项目8：<br>ICU死亡率预测<br>再入院风险评估 |
| 课时-12 | 自然语言处理在医疗中的应用<br>- 医学文本特征<br>- 命名实体识别(NER)<br>- 关系抽取<br>- 临床笔记分析<br>- 医学知识图谱 | 课件12 | ClinicalBERT论文<br>医学NLP综述<br>UMLS知识库<br>信息抽取工具 | 项目9：<br>疾病-症状抽取<br>医学问答系统 |
| 课时-13 | 药物发现中的计算方法<br>- 分子表示学习<br>- 图神经网络(GNN)<br>- 分子生成模型<br>- 药物-靶点相互作用预测 | 课件13 | 分子GNN综述<br>RDKit工具包<br>DeepChem框架<br>分子对接算法 | 项目10：<br>分子性质预测<br>药物相似性搜索 |
| 课时-14 | 大规模生物医学数据系统<br>- 分布式存储(HDFS/HBase)<br>- 并行计算框架(Spark/Dask)<br>- 工作流管理(Nextflow/Snakemake)<br>- 云计算平台应用 | 课件14 | 生物信息学工作流<br>AWS/Google Cloud案例<br>容器化部署(Docker/Singularity)<br>可重复性研究实践 | 项目11：<br>基因组分析流程<br>云端部署实践 |
| 课时-15 | 隐私保护与联邦学习<br>- 医疗数据隐私法规<br>- 差分隐私技术<br>- 联邦学习框架<br>- 同态加密与安全多方计算 | 课件15 | GDPR/HIPAA解读<br>联邦学习综述<br>PySyft框架<br>隐私保护案例研究 | 项目12：<br>联邦学习医疗应用<br>差分隐私实现 |
| 课时-16 | 前沿技术与课程总结<br>- 单细胞数据分析<br>- 空间转录组学<br>- 多模态学习<br>- 因果推断<br>- 课程项目展示 | 课件16 | 单细胞分析工具<br>多组学整合方法<br>因果推断框架<br>前沿会议论文(RECOMB/ISMB) | 期末项目：<br>综合应用开发<br>论文复现或创新 |

---

## 4. 编程工具与开发环境

### 必备工具
- **Python生态**: NumPy, Pandas, Scikit-learn, PyTorch/TensorFlow
- **生物信息学库**: Biopython, pysam, scikit-bio
- **可视化**: Matplotlib, Seaborn, Plotly
- **版本控制**: Git/GitHub

### 推荐平台
- **Jupyter Notebook**: 交互式开发
- **Google Colab**: 免费GPU资源
- **Docker**: 环境容器化
- **GitHub**: 代码管理与协作

---

## 5. 数据集与竞赛资源

### 公开数据集
- **基因组数据**: 1000 Genomes Project, TCGA, GTEx
- **医学影像**: ChestX-ray14, LIDC-IDRI, BraTS
- **临床数据**: MIMIC-III/IV, eICU
- **药物数据**: ChEMBL, PubChem, ZINC

### 竞赛平台
- **Kaggle**: 医疗健康竞赛专区
- **DREAM Challenges**: 计算生物学挑战赛
- **Grand Challenge**: 医学影像竞赛
- **阿里天池/DataFountain**: 国内医疗AI竞赛

---

## 6. 推荐在线课程与资源

| 课程名称 | 课程链接 |
|---------|---------|
| CMU计算生物学 | http://www.cs.cmu.edu/~02710/ |
| MIT计算生物学基础 | https://ocw.mit.edu/courses/biology/ |
| 斯坦福生物医学信息学 | https://med.stanford.edu/bmi.html |
| Coursera生物信息学专项课程 | https://www.coursera.org/specializations/bioinformatics |
| Rosalind生物信息学编程练习 | http://rosalind.info/ |
| Biostars生物信息学问答社区 | https://www.biostars.org/ |
| DeepLearning.AI医疗AI专项课程 | https://www.coursera.org/specializations/ai-for-medicine |
| Fast.ai深度学习课程 | https://www.fast.ai/ |

---

## 7. 开源项目推荐

- **Biopython**: Python生物信息学工具集
- **scikit-bio**: 生物信息学算法库
- **DeepChem**: 药物发现深度学习框架
- **Scanpy**: 单细胞数据分析
- **MONAI**: 医学影像深度学习框架
- **AlphaFold**: 蛋白质结构预测
- **OpenMM**: 分子动力学模拟

---

## 8. 主讲人学术资源

[计算方法在精准医疗中的应用], XXX学术期刊, 2023.

[从算法到临床：医疗AI的工程化实践], 中国计算机学会通讯, 2022.

[生物医学大数据的计算挑战与机遇], ACM通讯中文版, 2021.

**课程公众号/GitHub仓库**

[课程代码仓库] https://github.com/xxx/biomedical-informatics

[课程讨论区] https://github.com/xxx/biomedical-informatics/discussions

---

## 9. 联系方式

XXX 教授  
Email: xxx@xxx.edu.cn  
办公室: 计算机学院XXX室  
答疑时间: 每周X下午X:XX-X:XX

**助教团队**  
Email: xxx-ta@xxx.edu.cn

---

## 10. 课程考核方式

- **平时作业** (30%): 12次编程作业
- **课程项目** (40%): 期末综合项目(个人或小组)
- **论文阅读报告** (15%): 前沿论文精读与复现
- **课堂参与** (15%): 出勤、讨论、Quiz