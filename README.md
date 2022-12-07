## 人心脏细胞类型预测



### 运行环境构建

运行环境python版本>=3.8

所需依赖放置于requirements.txt中，运行下述命令安装所需依赖包：

```
cd celltype_predict
pip install -r requirements.txt
```

### 程序运行步骤

```
cd celltype_predict
# 预测level1
python model.py --input ./data/data_input.h5ad --levels first --output ../result/result_save.csv
# 预测level1-4
python model.py --input ./data/data_input.h5ad --levels all --output ../result/result_save_all.csv
```

### 结果输出



|                 cell_id |             level1 |             level2 |             level3 | level4             |
| ----------------------: | -----------------: | -----------------: | -----------------: | ------------------ |
| M1.1.AAAGGGCTCGCATAGT_1 | Cardiomyocyte cell | Cardiomyocyte cell | Cardiomyocyte cell | Cardiomyocyte cell |
| M1.1.AAAGGTAAGCTTAGTC_1 | Cardiomyocyte cell | Cardiomyocyte cell | Cardiomyocyte cell | Cardiomyocyte cell |
| M1.1.AACTTCTAGTTGCCTA_1 | Cardiomyocyte cell | Cardiomyocyte cell | Cardiomyocyte cell | Cardiomyocyte cell |
| M1.1.AATAGAGAGCGGATCA_1 | Cardiomyocyte cell | Cardiomyocyte cell | Cardiomyocyte cell | Cardiomyocyte cell |
| M1.1.ACGGTCGCACAATGAA_1 | Cardiomyocyte cell | Cardiomyocyte cell | Cardiomyocyte cell | Cardiomyocyte cell |



### 测试输出输入说明

以文件夹中测试数据为例，测试数据存放于data文件夹，测试结果存放于result文件夹。

输入数据和比赛所用数据需要有同样结构，obs中含有平台信息及donor信息

1. 将`所需预测数据`置于`data`文件夹下，重命名为`data_input.h5ad`，或通过--input参数指定文件所在路径
2. 运行`python model.py --input ./data/data_input.h5ad --levels first --output ../result/result_save.csv`得到level1的预测结果，返回结果保存于result文件夹
3. 运行`python model.py --input ./data/data_input.h5ad --levels all --output ../result/result_save_all.csv`得到level1-4的预测结果，返回结果保存于result文件夹