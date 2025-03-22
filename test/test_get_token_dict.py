from collections import defaultdict

# 定义一个默认值工厂函数，这里使用int来初始化为0
default_dict = defaultdict(int)

# 示例用法
print(default_dict['a'])  # 输出: 0，因为'a'不存在，自动初始化为0
default_dict['a'] += 1
print(default_dict['a'])  # 输出: 1

# 你也可以使用其他类型的工厂函数，例如list
list_default_dict = defaultdict(list)
print(list_default_dict['b'])  # 输出: []，因为'b'不存在，自动初始化为空列表
list_default_dict['b'].append(2)
print(list_default_dict['b'])  # 输出: [2]

# 如果你有一个现有的字典，也可以将其转换为defaultdict
existing_dict = {'c': 3, 'd': 4}
default_dict = defaultdict(int, existing_dict)
print(default_dict['c'])  # 输出: 3
print(default_dict['e'])  # 输出: 0，因为'e'不存在，自动初始化为0