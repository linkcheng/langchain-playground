import os
import ast
import ray
from ray import data as ds
from dotenv import load_dotenv

# 导入 tiktoken 库。Tiktoken 是 OpenAI 开发的一个库，用于从模型生成的文本中计算 token 数量。
import tiktoken
# 从 openai.embeddings_utils 包中导入 get_embedding 函数。
# 这个函数可以获取 GPT-3 模型生成的嵌入向量。
# 嵌入向量是模型内部用于表示输入数据的一种形式。
from openai.embeddings_utils import get_embedding, cosine_similarity


import numpy as np
# 从 matplotlib 包中导入 pyplot 子库，并将其别名设置为 plt。
# matplotlib 是一个 Python 的 2D 绘图库，pyplot 是其子库，提供了一种类似 MATLAB 的绘图框架。
import matplotlib.pyplot as plt
import matplotlib

# 从 sklearn.manifold 模块中导入 TSNE 类。
# TSNE (t-Distributed Stochastic Neighbor Embedding) 是一种用于数据可视化的降维方法，尤其擅长处理高维数据的可视化。
# 它可以将高维度的数据映射到 2D 或 3D 的空间中，以便我们可以直观地观察和理解数据的结构。
from sklearn.manifold import TSNE

# 从 scikit-learn中导入 KMeans 类。KMeans 是一个实现 K-Means 聚类算法的类。
from sklearn.cluster import KMeans


load_dotenv(verbose=True)
openai_api_key = os.getenv("openai_api_key")
ray_address = os.getenv("ray_address")

ray.init(address=ray_address, ignore_reinit_error=True)

TOP_N = 1000
MAX_TOKENS = 8000  

embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"


def load_data():
    input_datapath = "data/fine_food_reviews_1k.csv"
    data = ds.read_csv(input_datapath)
    return data


def transform(batch):
    # 将 "Summary" 和 "Text" 字段组合成新的字段 "combined"
    batch["combined"] = "Title: " + batch.Summary.str.strip() + "; Content: " + batch.Text.str.strip()
    batch = batch.sort_values("Time").tail(TOP_N * 2)
    batch.drop("Time", axis=1, inplace=True)
    
    encoding = tiktoken.get_encoding(embedding_encoding)
    batch["n_tokens"] = batch.combined.apply(lambda x: len(encoding.encode(x)))
    # 将样本减少到最近的 TOP_N 个评论，并删除过长的样本
    batch = batch[batch.n_tokens <= MAX_TOKENS].tail(TOP_N)
    return batch


def embedding(batch):
    # 实际生成会耗时几分钟
    batch["embedding"] = batch.combined.apply(lambda x: get_embedding(x, engine=embedding_model, api_key=openai_api_key))
    return batch


def ds_embedding(batch):
    batch.add_column("embedding", lambda df: df.combined.apply(lambda x: get_embedding(x, engine=embedding_model, api_key=openai_api_key)))


def embedding_vec(batch):
    batch["embedding_vec"] = batch.embedding.apply(ast.literal_eval)
    return batch


def save_embedding(data):
    output_datapath = "data/fine_food_reviews_with_embeddings_1k.csv"
    data.write_csv(output_datapath)


def load_embedding():
    embedding_datapath = "data/fine_food_reviews_with_embeddings_1k.csv"
    data = ds.read_csv(embedding_datapath)
    return data


def tsne_plot(df):
    colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
    # 将嵌入向量列表转换为二维 numpy 数组
    matrix = np.vstack(df["embedding_vec"].values)

    # 创建一个 t-SNE 模型，t-SNE 是一种非线性降维方法，常用于高维数据的可视化。
    # n_components 表示降维后的维度（在这里是2D）
    # perplexity 可以被理解为近邻的数量
    # random_state 是随机数生成器的种子
    # init 设置初始化方式
    # learning_rate 是学习率。
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    # 使用 t-SNE 对数据进行降维，得到每个数据点在新的2D空间中的坐标
    vis_dims = tsne.fit_transform(matrix)
    
    # 从降维后的坐标中分别获取所有数据点的横坐标和纵坐标
    x = [_x for _x, _ in vis_dims]
    y = [_y for _, _y in vis_dims]

    # 根据数据点的评分（减1是因为评分是从1开始的，而颜色索引是从0开始的）获取对应的颜色索引
    color_indices = df.Score.values - 1
    # 确保你的数据点和颜色索引的数量匹配
    assert len(vis_dims) == len(df.Score.values)

    # 创建一个基于预定义颜色的颜色映射对象
    colormap = matplotlib.colors.ListedColormap(colors)
    # 使用 matplotlib 创建散点图，其中颜色由颜色映射对象和颜色索引共同决定，alpha 是点的透明度
    plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
    # 为图形添加标题
    plt.title("Amazon ratings visualized in language using t-SNE")
    plt.show()


def kmean_plot(df):
    # 定义要生成的聚类数。
    n_clusters = 4
    matrix = np.vstack(df["embedding_vec"].values)
    
    # 创建一个 KMeans 对象，用于进行 K-Means 聚类。
    # n_clusters 参数指定了要创建的聚类的数量；
    # init 参数指定了初始化方法（在这种情况下是 'k-means++'）；
    # random_state 参数为随机数生成器设定了种子值，用于生成初始聚类中心。
    kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)

    # 使用 matrix（我们之前创建的矩阵）来训练 KMeans 模型。这将执行 K-Means 聚类算法。
    kmeans.fit(matrix)

    # kmeans.labels_ 属性包含每个输入数据点所属的聚类的索引。
    # 这里，我们创建一个新的 'Cluster' 列，在这个列中，每个数据点都被赋予其所属的聚类的标签。
    df['Cluster'] = kmeans.labels_
    
    # 首先为每个聚类定义一个颜色。
    colors = ["red", "green", "blue", "purple"]

    # 然后，你可以使用 t-SNE 来降维数据。这里，我们只考虑 'embedding_vec' 列。
    tsne_model = TSNE(n_components=2, random_state=42)
    vis_data = tsne_model.fit_transform(matrix)

    # 现在，你可以从降维后的数据中获取 x 和 y 坐标。
    x = vis_data[:, 0]
    y = vis_data[:, 1]

    # 'Cluster' 列中的值将被用作颜色索引。
    color_indices = df['Cluster'].values

    # 创建一个基于预定义颜色的颜色映射对象
    colormap = matplotlib.colors.ListedColormap(colors)

    # 使用 matplotlib 创建散点图，其中颜色由颜色映射对象和颜色索引共同决定
    plt.scatter(x, y, c=color_indices, cmap=colormap)
    # 为图形添加标题
    plt.title("Clustering visualized in 2D using t-SNE")
    # 显示图形
    plt.show()


def search_reviews(df, product_description, n=3, pprint=True):
    # 定义一个名为 search_reviews 的函数，
    # Pandas DataFrame 产品描述，数量，以及一个 pprint 标志（默认值为 True）。
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002",
        api_key=openai_api_key,
    )
    df["similarity"] = df.embedding_vec.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results


if __name__ == '__main__':
    data_set = load_data()
    # data_batch = data_set.take_batch(10, batch_format="pandas")
    data_set1 = data_set.map_batches(transform, batch_format="pandas")
    # data_set2 = data_set1.map_batches(embedding, batch_format="pandas")
    # save_embedding(data_set2)
    
    df_embedded = load_embedding().to_pandas()
    df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)
    
    tsne_plot(df_embedded) 
    kmean_plot(df_embedded)
    
    res = search_reviews(df_embedded, df_embedded["combined"][0], n=3)
    print(res)
    res = search_reviews(df_embedded, "awful", n=3)
    print(res)
