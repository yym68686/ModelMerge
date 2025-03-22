def run_python_script(script):
    # 创建一个字典来存储脚本执行的本地变量
    local_vars = {}

    try:
        # 执行脚本字符串
        exec(script, {}, local_vars)
        return local_vars
    except Exception as e:
        return str(e)

# 示例用法
script = "# \u8ba1\u7b97\u524d100\u4e2a\u6590\u6ce2\u7eb3\u5207\u6570\u5217\u7684\u548c\n\ndef fibonacci_sum(n):\n    a, b = 0, 1\n    sum = 0\n    for _ in range(n):\n        sum += a\n        a, b = b, a + b\n    return sum\n\nfibonacci_sum(100)"
print(script)
output = run_python_script(script)
print(output)
# 下面是要运行的程序，怎么修改上面的代码，可以捕获fibonacci_sum的输出
def fibonacci_sum(n):
    a, b = 0, 1
    sum = 0
    for _ in range(n):
        sum += a
        a, b = b, a + b
    return sum

print(fibonacci_sum(100))