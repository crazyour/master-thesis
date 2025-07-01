import os
import torch
import rlcard
import random
import copy
import shutil
   
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from PIL import UnidentifiedImageError
from torchvision.models import ResNet50_Weights
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device,flush=True)
# 初始化 ResNet50 模型的函数
def initialize_resnet50():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # 假设十分类
    return model
def evaluate(model_path):
    # 数据预处理，与训练时保持一致
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 自定义数据集类来处理损坏的图像文件
    class VerifiedImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            try:
                img = self.loader(path)
                if self.transform is not None:
                    img = self.transform(img)
                return img, target
            except (UnidentifiedImageError, OSError):
                return None  # 返回 None 以跳过此项

    # 加载测试数据集
    test_dataset = VerifiedImageFolder(root='/home/luanma12/recognition_10/data/train', transform=transform)

    # 创建自定义 DataLoader 以过滤掉 None
    def collate_fn(batch):
        # 过滤掉 None 条目
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 加载训练好的模型
    model = initialize_resnet50()  # 初始化与保存时一致的模型结构
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载权重到模型中
    model = model.to(device)  # 将模型移动到 GPU 或 CPU 上
    model.eval()  # 设置模型为评估模式

    # 定义损失函数（可选）
    criterion = nn.CrossEntropyLoss()

    # 测试模型性能
    correct = 0
    total = 0
    test_loss = 0.0

    # 进行推理
    with torch.no_grad():
        for images, labels in test_loader:
            # 将数据移动到与模型一致的设备上
            images, labels = images.to(device), labels.to(device)
            # 前向传播得到输出
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算测试集的准确率和平均损失
    accuracy = round(correct / total, 3)
    return accuracy


def clear_model(path): # 指定されたパスの下のlandlord, landlord_up, landlord_downモデルをクリアする
    if os.path.exists(path):
        # ディレクトリ内のすべてのファイルをループする
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            os.unlink(file_path)  # ファイルまたはシンボリックリンクを削除する
def copy_model(from_path, to_path):  
    for filename in os.listdir(from_path):
        src_file = os.path.join(from_path, filename)
        dest_file = os.path.join(to_path, filename)  
        shutil.copy2(src_file, dest_file)  # ファイルをコピーする


def fitness(population_pool, fitness_list,population_type):
   
    for i, model in enumerate(population_pool[population_type]):

        # 计算模型奖励
        reward =evaluate(model)  # 假设 calculate_reward 是计算奖励的函数

        # 将奖励存放在 fitness_list 中的相应位置
        fitness_list[population_type][i] = reward
            
    return fitness_list

def calculate_l2_distance(model_0_path, model_1_path):
    # 直接加载参数字典
    model_0_params = torch.load(model_0_path, device,weights_only=True)
    model_1_params = torch.load(model_1_path, device,weights_only=True)
    
    # 计算 L2 距离
    l2_distance = 0.0
    for name in model_0_params:
        if name in model_1_params:
            param_diff = model_0_params[name] - model_1_params[name]
            l2_distance += torch.sum(param_diff ** 2).item()
    
    return round(l2_distance ** 0.5, 2)


def crossover(children_population_size,population_pool):
     # 1. 生成所有合法组合
    all_combinations = [(i, j) for i in range(parent_population_size) for j in range(parent_population_size) if i != j]

    # 2. 从所有合法组合中随机抽取所需数量的不重复组合
    selected_combinations = random.sample(all_combinations, children_population_size // 2)
    model_0 = initialize_resnet50()
    model_1 = initialize_resnet50()
    #精英种群交叉
    for elite_idx, (model0_idx, model1_idx) in enumerate(selected_combinations): 
        model0_path = population_pool['parents_elite'][model0_idx]
        model1_path = population_pool['parents_elite'][model1_idx]
        # 加载已有的两个 DQN 模型

        model_0.load_state_dict(torch.load(model0_path, device,weights_only=True))
        model_1.load_state_dict(torch.load(model1_path, device,weights_only=True))

        model_2_params = {}

        for name in model_0.state_dict().keys():
            model_2_params[name] = model_0.state_dict()[name] if random.random() < 0.5 else model_1.state_dict()[name]
                
        # 设置变异率和变异范围
        mutation_rate = 0.2  # 变异概率
        mutation_range = 0.1  # 变异范围
        # 对 model_2 的参数进行变异
        for name, param in model_2_params.items():
            if random.random() < mutation_rate and param.dtype.is_floating_point:
                noise = (torch.rand_like(param) - 0.5) * 2 * mutation_range
                param += param * noise

        # 保存包含完整结构和新参数的 model_2
        torch.save(model_2_params, f"/home/luanma12/recognition_10/evolutionary/model/children_population/model_{elite_idx}.pth")
        population_pool['children'][elite_idx]=f"/home/luanma12/recognition_10/evolutionary/model/children_population/model_{elite_idx}.pth"
        
    #精英种群和多样性种群交叉    
   # 1. 生成所有组合（包括相同数字的组合）
    all_combinations = [(i, j) for i in range(parent_population_size) for j in range(parent_population_size)]

    # 2. 从所有组合中随机抽取所需数量的不重复组合
    selected_combinations = random.sample(all_combinations,(children_population_size- children_population_size // 2))

    for elite_diverse_idx, (model0_idx, model1_idx) in enumerate(selected_combinations):
        elite_diverse_idx=elite_diverse_idx+int(children_population_size/2)
        model0_path = population_pool['parents_elite'][model0_idx]
        model1_path = population_pool['parents_diverse'][model1_idx]
        # 加载已有的两个 DQN 模型
        model_0.load_state_dict(torch.load(model0_path, device,weights_only=True))
        model_1.load_state_dict(torch.load(model1_path, device,weights_only=True))
       
        model_2_params = {}

        for name in model_0.state_dict().keys():
            model_2_params[name] = model_0.state_dict()[name] if random.random() < 0.5 else model_1.state_dict()[name]
                
        # 设置变异率和变异范围
        mutation_rate = 0.2  # 变异概率
        mutation_range = 0.1  # 变异范围
        # 对 model_2 的参数进行变异
        for name, param in model_2_params.items():
            if random.random() < mutation_rate and param.dtype.is_floating_point:
                noise = (torch.rand_like(param) - 0.5) * 2 * mutation_range
                param += param * noise

        # 保存包含完整结构和新参数的 model_2
        torch.save(model_2_params, f"/home/luanma12/recognition_10/evolutionary/model/children_population/model_{elite_diverse_idx}.pth")
        population_pool['children'][elite_diverse_idx]=f"/home/luanma12/recognition_10/evolutionary/model/children_population/model_{elite_diverse_idx}.pth"
    return population_pool

def eliminate(population_pool, fitness_list):
    population_pool_temporary = {'parents_elite':[0]*parent_population_size,'parents_diverse':[0]*parent_population_size,'children':[0]*children_population_size}
    fitness_list_temporary = {'parents_elite':[0]*parent_population_size,'parents_diverse':[0]*parent_population_size,'children':[0]*children_population_size}
    similarity={'parents_diverse':[0]*parent_population_size,'children':[0]*children_population_size}
    
    combined_indices = {}
    # 親と子の適応値を組み合わせて、適応値の高い順にソートし、上位4個体を選択する
    combined_scores = [(score, 'parents_elite', i) for i, score in enumerate(fitness_list['parents_elite'])] + \
                          [(score, 'parents_diverse', i) for i, score in enumerate(fitness_list['parents_diverse'])]+ \
                              [(score, 'children', i) for i, score in enumerate(fitness_list['children'])]
    sorted_scores = sorted(combined_scores, key=lambda x: x[0], reverse=True)[:parent_population_size]
    combined_indices= sorted_scores
    
    # 上位のモデルを parent_population_temporary に保存し、次の世代の親とする
    for i, combination in zip(range(parent_population_size), combined_indices):
        population_pool_temporary['parents_elite'][i]=f"/home/luanma12/recognition_10/evolutionary/model/parent_population_elite/model_{i}.pth"
        fitness_list_temporary['parents_elite'][i] = fitness_list[combination[1]][combination[2]]
        shutil.copy(population_pool[combination[1]][combination[2]], f"/home/luanma12/recognition_10/evolutionary/model/parent_population_temporary/elite/model_{i}.pth")
        
    #计算子代的相似度
    model=population_pool[combined_indices[0][1]][combined_indices[0][2]]#适应度最高的模型
    for index in range(children_population_size):
        similarity['children'][index]=calculate_l2_distance(model,f"/home/luanma12/recognition_10/evolutionary/model/children_population/model_{index}.pth")
    for index in range(parent_population_size):
        similarity['parents_diverse'][index]=calculate_l2_distance(model,f"/home/luanma12/recognition_10/evolutionary/model/parent_population_diverse/model_{index}.pth")
    # 定义 combined_indices 用于存储符合条件的 (source, index)
    combined_indices = []
    # 大于10的项
    greater_than_10 = []
    # 小于或等于10的项
    less_than_10 = []
    # 遍历 similarity['children']，根据值大小分别保存
    for index, value in enumerate(similarity['children']):
        if value > 10:
            greater_than_10.append(('children', index))
        else:
            less_than_10.append(('children', index))

    # 遍历 similarity['parents_diverse']，根据值大小分别保存
    for index, value in enumerate(similarity['parents_diverse']):
        if value > 10:
            greater_than_10.append(('parents_diverse', index))
        else:
            less_than_10.append(('parents_diverse', index))

    # 如果大于10的项足够
    if len(greater_than_10) >= parent_population_size:
        random_selected_indices = random.sample(greater_than_10, k=parent_population_size)
    else:
        # 如果不够，则从小于或等于10的项中补齐
        random_selected_indices = greater_than_10
        remaining_needed = parent_population_size - len(greater_than_10)
        random_selected_indices += random.sample(less_than_10, k=remaining_needed)
    
    # 上位のモデルを parent_population_temporary に保存し、次の世代の親とする
    for i, combination in zip(range(parent_population_size), random_selected_indices):
        population_pool_temporary['parents_diverse'][i]=f"/home/luanma12/recognition_10/evolutionary/model/parent_population_diverse/model_{i}.pth"
        fitness_list_temporary['parents_diverse'][i] = fitness_list[combination[0]][combination[1]]
        shutil.copy(population_pool[combination[0]][combination[1]], f"/home/luanma12/recognition_10/evolutionary/model/parent_population_temporary/diverse/model_{i}.pth")
    
    # children_population フォルダをクリア
    clear_model('/home/luanma12/recognition_10/evolutionary/model/children_population')

    # parent_population フォルダをクリア
    clear_model('/home/luanma12/recognition_10/evolutionary/model/parent_population_elite')
    clear_model('/home/luanma12/recognition_10/evolutionary/model/parent_population_diverse')
    
    # temporary_population のすべてのフォルダを parent_population にコピー
    copy_model("/home/luanma12/recognition_10/evolutionary/model/parent_population_temporary/elite","/home/luanma12/recognition_10/evolutionary/model/parent_population_elite")
    copy_model("/home/luanma12/recognition_10/evolutionary/model/parent_population_temporary/diverse","/home/luanma12/recognition_10/evolutionary/model/parent_population_diverse")
    # temporary_population フォルダをクリア
    clear_model('/home/luanma12/recognition_10/evolutionary/model/parent_population_temporary/elite')
    clear_model('/home/luanma12/recognition_10/evolutionary/model/parent_population_temporary/diverse')
    
    population_pool = population_pool_temporary
    fitness_list = fitness_list_temporary
    print("精英",fitness_list['parents_elite'], flush=True)
    print("多样性",fitness_list['parents_diverse'], flush=True)
    print("\n",flush=True)
    return population_pool, fitness_list

def initialize_population(parent_population_size, population_pool,fitness_list):
    # 初期化アドレスデータ
    for i in range(parent_population_size):
        population_pool['parents_elite'][i]=f"/home/luanma12/recognition_10/evolutionary/model/parent_population_elite/model_{i}.pth"
        population_pool['parents_diverse'][i]=f"/home/luanma12/recognition_10/evolutionary/model/parent_population_diverse/model_{i}.pth"
    # 初期化ファイルデータ
    clear_model('/home/luanma12/recognition_10/evolutionary/model/children_population')
    clear_model('/home/luanma12/recognition_10/evolutionary/model/parent_population_elite')
    clear_model('/home/luanma12/recognition_10/evolutionary/model/parent_population_diverse')
    #计算初始模型的适应值
    fitness=[0]*2*parent_population_size
    for index in range(2*parent_population_size):
        model = f"/home/luanma12/recognition_10/evolutionary/model/rl_model/model_{index}.pth"
        # 计算模型奖励
        reward =evaluate(model)  # 假设 calculate_reward 是计算奖励的函数
        # 将奖励存放在 fitness_list 中的相应位置
        fitness[index] = reward

    #将fitness从大到小排序，记录fitness中的索引以及值
    sorted_fitness= sorted(enumerate(fitness), key=lambda x: x[1], reverse=True)
    #将适应值排前面一半的模型复制到parent_population_elite以及记录它们的适应值到fitness_list中
    for idx, (original_index, fitness_value) in enumerate(sorted_fitness):
        if idx==parent_population_size:
            break
        shutil.copy2(f"/home/luanma12/recognition_10/evolutionary/model/rl_model/model_{original_index}.pth",f"/home/luanma12/recognition_10/evolutionary/model/parent_population_elite/model_{idx}.pth")  # ファイルをコピーする
        fitness_list['parents_elite'][idx]=fitness_value
    
    
    #计算所有模型和适应值最高的模型的相似度
    similarity=[0]*2*parent_population_size
    model=f"/home/luanma12/recognition_10/evolutionary/model/parent_population_elite/model_{0}.pth"
    for index in range(2*parent_population_size):
        similarity[index]=calculate_l2_distance(model,f"/home/luanma12/recognition_10/evolutionary/model/rl_model/model_{index}.pth")
    sorted_similarity= sorted(enumerate(similarity), key=lambda x: x[1], reverse=True)
    #将相似度最低(值越大相似度越低)排前面一半的模型复制到parent_population_elite以及记录它们的适应值到fitness_list中
    for idx, (original_index, similarity_value) in enumerate(sorted_similarity):
        if idx==parent_population_size:
            break
        shutil.copy2(f"/home/luanma12/recognition_10/evolutionary/model/rl_model/model_{original_index}.pth",f"/home/luanma12/recognition_10/evolutionary/model/parent_population_diverse/model_{idx}.pth")  # ファイルをコピーする
        fitness_list['parents_diverse'][idx]=fitness[original_index]
    return population_pool,fitness_list

generations = 500
parent_population_size = 8
children_population_size = 30
population_pool = {'parents_elite':[0]*parent_population_size,'parents_diverse':[0]*parent_population_size,'children':[0]*children_population_size}
fitness_list = {'parents_elite':[0]*parent_population_size,'parents_diverse':[0]*parent_population_size,'children':[0]*children_population_size}
population_pool,fitness_list = initialize_population(parent_population_size, population_pool,fitness_list)
print("精英:", fitness_list['parents_elite'],flush=True)
print("多样性", fitness_list['parents_diverse'],flush=True)



for generation in range(generations):
    print(f"第{generation+1}世代:",flush=True)
    population_pool=crossover(children_population_size,population_pool)
    fitness_list = fitness(population_pool, fitness_list, "children")
    population_pool, fitness_list = eliminate(population_pool, fitness_list)








# import os
# import torch
# import random
# import shutil
# import torch.nn as nn
# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader
# import torchvision.models as models
# from PIL import UnidentifiedImageError

# # 设备配置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 初始化 ResNet50 模型的函数
# def initialize_resnet50():
#     model = models.resnet50(weights=None)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, 10)  # 假设为10分类
#     return model

# # 评估函数
# def evaluate(model_path):
#     # 数据预处理，与训练时保持一致
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     # 自定义数据集类来处理损坏的图像文件
#     class VerifiedImageFolder(datasets.ImageFolder):
#         def __getitem__(self, index):
#             path, target = self.samples[index]
#             try:
#                 img = self.loader(path)
#                 if self.transform is not None:
#                     img = self.transform(img)
#                 return img, target
#             except (UnidentifiedImageError, OSError):
#                 return None  # 跳过损坏图像

#     # 加载测试数据集
#     test_dataset = VerifiedImageFolder(root='/home/luanma12/recognition_10/data/train', transform=transform)

#     # 自定义 DataLoader，过滤 None
#     def collate_fn(batch):
#         batch = list(filter(lambda x: x is not None, batch))
#         return torch.utils.data.dataloader.default_collate(batch)

#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

#     # 加载模型
#     model = initialize_resnet50()
#     print(model_path)
#     try:
#         model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
#     except RuntimeError as e:
#         print(f"加载模型权重失败: {e}")
#         return 0.0
#     model = model.to(device)
#     model.eval()

#     # 定义损失函数（可选）
#     criterion = nn.CrossEntropyLoss()

#     # 测试模型性能
#     correct = 0
#     total = 0
#     test_loss = 0.0

#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             test_loss += loss.item()

#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = round(correct / total, 3)
#     return accuracy

# # 计算 L2 距离函数
# def calculate_l2_distance(model_0_path, model_1_path):
#     model_0 = initialize_resnet50()
#     model_1 = initialize_resnet50()

#     model_0.load_state_dict(torch.load(model_0_path, map_location=device))
#     model_1.load_state_dict(torch.load(model_1_path, map_location=device))

#     l2_distance = 0.0
#     for name, param_0 in model_0.state_dict().items():
#         if name in model_1.state_dict():
#             param_1 = model_1.state_dict()[name]
#             l2_distance += torch.sum((param_0 - param_1) ** 2).item()
#     return round(l2_distance ** 0.5, 2)

# # 交叉函数
# def crossover(children_population_size,population_pool):
#      # 1. 生成所有合法组合
#     all_combinations = [(i, j) for i in range(parent_population_size) for j in range(parent_population_size) if i != j]

#     # 2. 从所有合法组合中随机抽取所需数量的不重复组合
#     selected_combinations = random.sample(all_combinations, children_population_size // 6)
#     model_0 = initialize_resnet50()
#     model_1 = initialize_resnet50()
#     #精英种群交叉
#     for elite_idx, (model0_idx, model1_idx) in enumerate(selected_combinations): 
#         model0_path = population_pool['parents_elite'][model0_idx]
#         model1_path = population_pool['parents_elite'][model1_idx]
#         # 加载已有的两个 DQN 模型

#         model_0.load_state_dict(torch.load(model0_path, device,weights_only=True))
#         model_1.load_state_dict(torch.load(model1_path, device,weights_only=True))

#         model_2_params = {}

#         for name in model_0.state_dict().keys():
#             model_2_params[name] = model_0.state_dict()[name] if random.random() < 0.5 else model_1.state_dict()[name]
                
#         # 设置变异率和变异范围
#         mutation_rate = 0.2  # 变异概率
#         mutation_range = 0.1  # 变异范围
#         # 对 model_2 的参数进行变异
#         for name, param in model_2_params.items():
#             if random.random() < mutation_rate and param.dtype.is_floating_point:
#                 noise = (torch.rand_like(param) - 0.5) * 2 * mutation_range
#                 param += param * noise

#         # 保存包含完整结构和新参数的 model_2
#         torch.save(model_2_params, f"/home/luanma12/recognition_10/evolutionary/model/children_population/model_{elite_idx}.pth")
#         population_pool['children'][elite_idx]=f"/home/luanma12/recognition_10/evolutionary/model/children_population/model_{elite_idx}.pth"
        
#     #精英种群和多样性种群交叉    
#    # 1. 生成所有组合（包括相同数字的组合）
#     all_combinations = [(i, j) for i in range(parent_population_size) for j in range(parent_population_size)]

#     # 2. 从所有组合中随机抽取所需数量的不重复组合
#     selected_combinations = random.sample(all_combinations,(children_population_size- children_population_size // 6))

#     for elite_diverse_idx, (model0_idx, model1_idx) in enumerate(selected_combinations): 
#         elite_diverse_idx=elite_diverse_idx+int(children_population_size/2)
#         model0_path = population_pool['parents_elite'][model0_idx]
#         model1_path = population_pool['parents_diverse'][model1_idx]
#         # 加载已有的两个 DQN 模型
#         model_0.load_state_dict(torch.load(model0_path, device,weights_only=True))
#         model_1.load_state_dict(torch.load(model1_path, device,weights_only=True))
       
#         model_2_params = {}

#         for name in model_0.state_dict().keys():
#             model_2_params[name] = model_0.state_dict()[name] if random.random() < 0.5 else model_1.state_dict()[name]
                
#         # 设置变异率和变异范围
#         mutation_rate = 0.2  # 变异概率
#         mutation_range = 0.1  # 变异范围
#         # 对 model_2 的参数进行变异
#         for name, param in model_2_params.items():
#             if random.random() < mutation_rate and param.dtype.is_floating_point:
#                 noise = (torch.rand_like(param) - 0.5) * 2 * mutation_range
#                 param += param * noise

#         # 保存包含完整结构和新参数的 model_2
#         torch.save(model_2_params, f"/home/luanma12/recognition_10/evolutionary/model/children_population/model_{elite_diverse_idx}.pth")
#         population_pool['children'][elite_diverse_idx]=f"/home/luanma12/recognition_10/evolutionary/model/children_population/model_{elite_diverse_idx}.pth"
#     return population_pool

# # 清理和复制函数
# def clear_model(path):
#     if os.path.exists(path):
#         for filename in os.listdir(path):
#             file_path = os.path.join(path, filename)
#             os.unlink(file_path)

# def copy_model(from_path, to_path):
#     for filename in os.listdir(from_path):
#         src_file = os.path.join(from_path, filename)
#         dest_file = os.path.join(to_path, filename)
#         shutil.copy2(src_file, dest_file)

# # 初始化种群
# def initialize_population(parent_population_size, population_pool, fitness_list):
#     for i in range(parent_population_size):
#         population_pool['parents_elite'][i] = f"/home/luanma12/recognition_10/evolutionary/model/parent_population_elite/model_{i}.pth"
#         population_pool['parents_diverse'][i] = f"/home/luanma12/recognition_10/evolutionary/model/parent_population_diverse/model_{i}.pth"

#     clear_model('/home/luanma12/recognition_10/evolutionary/model/children_population')
#     clear_model('/home/luanma12/recognition_10/evolutionary/model/parent_population_elite')
#     clear_model('/home/luanma12/recognition_10/evolutionary/model/parent_population_diverse')

#     # 初始化适应值
#     fitness = [0] * (2 * parent_population_size)
#     for idx in range(2 * parent_population_size):
#         model = f"/home/luanma12/recognition_10/evolutionary/model/rl_model/model_{idx}.pth"
#         reward = evaluate(model)
#         fitness[idx] = reward

#     sorted_fitness = sorted(enumerate(fitness), key=lambda x: x[1], reverse=True)

#     for idx, (original_index, fitness_value) in enumerate(sorted_fitness[:parent_population_size]):
#         shutil.copy2(f"/home/luanma12/recognition_10/evolutionary/model/rl_model/model_{original_index}.pth",
#                      f"/home/luanma12/recognition_10/evolutionary/model/parent_population_elite/model_{idx}.pth")
#         fitness_list['parents_elite'][idx] = fitness_value

#     return population_pool, fitness_list

# # 适应值计算
# def fitness(population_pool, fitness_list, population_type):
#     for i, model_path in enumerate(population_pool[population_type]):
#         fitness_list[population_type][i] = evaluate(model_path)
#     return fitness_list

# # 淘汰函数
# def eliminate(population_pool, fitness_list):
#     population_pool_temporary = {'parents_elite': [0] * parent_population_size,
#                                  'parents_diverse': [0] * parent_population_size,
#                                  'children': [0] * children_population_size}
#     fitness_list_temporary = {'parents_elite': [0] * parent_population_size,
#                                'parents_diverse': [0] * parent_population_size,
#                                'children': [0] * children_population_size}

#     combined_scores = [(score, 'parents_elite', i) for i, score in enumerate(fitness_list['parents_elite'])] + \
#                       [(score, 'children', i) for i, score in enumerate(fitness_list['children'])]
#     sorted_scores = sorted(combined_scores, key=lambda x: x[0], reverse=True)[:parent_population_size]

#     for idx, (_, source, i) in enumerate(sorted_scores):
#         model_path = population_pool[source][i]
#         population_pool_temporary['parents_elite'][idx] = model_path
#         fitness_list_temporary['parents_elite'][idx] = fitness_list[source][i]

#     return population_pool_temporary, fitness_list_temporary

# # 主流程
# generations = 100
# parent_population_size = 8
# children_population_size = 30
# population_pool = {'parents_elite': [0] * parent_population_size,
#                    'parents_diverse': [0] * parent_population_size,
#                    'children': [0] * children_population_size}
# fitness_list = {'parents_elite': [0] * parent_population_size,
#                 'parents_diverse': [0] * parent_population_size,
#                 'children': [0] * children_population_size}

# population_pool, fitness_list = initialize_population(parent_population_size, population_pool, fitness_list)
# print("初始精英:", fitness_list['parents_elite'], flush=True)
# print(population_pool)
# for generation in range(generations):
#     print(f"第 {generation + 1} 世代:", flush=True)
#     population_pool = crossover(children_population_size, population_pool)
#     fitness_list = fitness(population_pool, fitness_list, 'children')
#     population_pool, fitness_list = eliminate(population_pool, fitness_list)
#     print(f"精英适应值: {fitness_list['parents_elite']}", flush=True)
