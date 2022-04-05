# Pytorch载入导出、冻结权重模板大全

## Ⅰ.写在前面

​	本文主要总结几类常用的Pytorch训练模型权重的载入导出及冻结的模板，方便各位即时翻看修改调用。注意，此模板仅适用于最普通的Pytorch训练参数权重，目前大部分论文给出的预训练权重模型都不能直接套用下面模板，需要稍加调整，但导入导出冻结的原理是一致的。



*参考资料：[Facebook-Slowfast](https://github.com/facebookresearch/SlowFast)

[TOC]

------



## Ⅱ.载入导出

### 一、载入

​	一般来说，利用Pytorch框架构建的模型都是直接继承自nn.Module，大致结构如下：

```python
class Your_Model(nn.Module):

	def __init__(self):
		super().__init__()
		
	def forward(self,x):
		return x
    
model = Your_Model(...)
```

​	我们可以通过：

```python
for p in model.named_parameters():
	print(p[0])
```

​	打印模型中的各层权重，下面为在Mvit模型中打印的权重：

> cls_token
>
> pos_embed
>
> patch_embed.proj.weight
>
> patch_embed.proj.bias
>
> blocks.0.norm1.weight
>
> blocks.0.norm1.bias
>
> ......

​	因此，若需要载入权重，直接通过：

```python
model.load_state_dict(pre_train_dict_match：, strict=False) #pre_train_dict_match ： dictionary
```

​	即可直接将pre_train_dict_match中的权重导入模型中。

#### ①模板一：只载入匹配维度的权重

​	顾名思义，只将预训练权重中与当前模型维度匹配的权重导入，其他的维度不匹配直接遗弃（不然因为维度不匹配也不能直接导入呀）。

```python
#导入预训练权重到字典中
state_dict = torch.load('path_to_your_checkpoint')
#导入模型权重到字典中
model_dict = model.state_dict()

#将所有预训练权重维度和现有模型权重匹配的记录在pre_train_dict_match中
pre_train_dict_match = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }

#将pre_train_dict_match中的权重导入模型
model.load_state_dict(pre_train_dict_match, strict=False)
```

#### ②模板二：只载入前几层权重

```python
#定义导入前N层权重
N = 10
#导入预训练权重到字典中
state_dict = torch.load('path_to_your_checkpoint')

#将前N层权重记录在pre_train_dict_load中
pre_train_dict_load = {
            k: v
    		for (_, (k, v)) in zip(range(len(pre_train_dict)), pre_train_dict.items())
            if _< N	#仅导入前N层权重
        }

#将pre_train_dict_match中的权重导入模型
model.load_state_dict(pre_train_dict_load, strict=False)
```



若还需要载入optimizer和scaler的权重，同理如上通过load_state_dict导入：

```python
if optimizer:
	optimizer.load_state_dict(load_dict)
if scaler:
    scaler.load_state_dict(load_dict)
```



### 二、导出权重

#### ①模板一：只导出模型参数

```python
if acc > best_acc:
	torch.save(model.state_dict(), f'{weights_path}/epoch{epoch_id}-{acc}.pth')
 	best_acc = acc	#只导出与记录训练效果最好的权重
```

​	直接用torch.save即可



#### ②模板二：导出所有权重

```python
#创建checkpoint字典存储各参数
checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
#若有scaler则记录
if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()

#导出权重
with open(path_to_checkpoint, "wb") as f:
	torch.save(checkpoint, f)
```

​	写成嵌套字典的方式导出权重，若需导入此文件则需修改对应导入代码，如下为optimizer和scaler的导入方式

```python
if optimizer:
	optimizer.load_state_dict(checkpoint["optimizer_state"])
if scaler:
	scaler.load_state_dict(checkpoint["scaler_state"])
```



------



## Ⅲ.冻结参数

### 一、冻结原理

​	若你的模型依旧是继承自nn.Module构建的，那么冻结该层权重不参与模型训练是通过设置权重的requires_grad属性为False来实现的。

```python
model.weight.requires_grad = False
```

​	同时，告知optimizer这层已经被冻结了，优化时可以直接跳过提升效率。（实测大部分情况直接设置requires_grad属性为False即可，无需告知optimizer，只不过如此会降低优化效率而已。）

```python
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
```



### 二、冻结模板

#### ①模板一：按照层数冻结权重

```python
#需要冻结的层数
indexs = [0,1,2]

for idx, para in enumerate(model.named_parameters()):
	if idx not in indexs:
		continue
	else:
		para[1].requires_grad = False #只冻结在indexs内的层数
```

#### ②模板二：按照层名冻结权重

```python
for p in model.named_parameters():
    if p[0] in frozen_list: #只冻结在名字列表内的权重
    	p[1].requires_grad = False
```

#### ③模板三：导出冻结层数名

```python
#将需要冻结的层提取出来
base_dir = path.dirname(path.abspath(__file__))
FL = dict.fromkeys(need_frozen_layers.keys(), 0)

#写出为json文件方便下次训练重新冻结
with open(base_dir + '/frozen_layers.json', 'w') as f:
	json.dump(FL, f)
```

#### ④模板四：显示当前模型参数冻结情况

```python
for p in model.named_parameters():
    print(f"{p[0]}'s requires_grad is {p[1].requires_grad}")
```

实际运行情况如下：

> cls_token's requires_grad is False
>
> pos_embed's requires_grad is True
>
> patch_embed.proj.weight's requires_grad is False
>
> patch_embed.proj.bias's requires_grad is False
>
> blocks.0.norm1.weight's requires_grad is False
>
> blocks.0.norm1.bias's requires_grad is False
>
> blocks.0.attn.q.weight's requires_grad is True
>
> blocks.0.attn.q.bias's requires_grad is True
>
> ......

------

以上代码均测试有效，若有问题欢迎讨论。