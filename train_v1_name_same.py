from PIL import Image
from augly.image.transforms import *
import torchvision.transforms as transforms
import random
import argparse
import pandas as pd

class ToRGB:
    def __call__(self, x):
        return x.convert("RGB")

class RandomResizeCrop:
    def __call__(self, x):
        tran = transforms.RandomResizedCrop(256, scale=(0.3,1))
        return tran(x)

class RandomBlur:
    def __init__(self, radius = [2, 5], name = 'RandomBlur'):
        self.radius = radius
        self.name = name
    
    def __call__(self, x):
        radius = random.uniform(self.radius[0], self.radius[1])
        x = Blur(radius = radius)(x)
        return x
    
class RandomSaturation:
    def __init__(self, factors = [2, 10], name = 'RandomSaturation'):
        self.factors = factors
        self.name = name
    
    def __call__(self, x):
        factor = random.uniform(self.factors[0], self.factors[1])
        x = Saturation(factor = factor)(x)
        return x
    
class RandomOverlayText(object):
    def __init__(self, num = 2, text = [0,20], color_1=[0,255], color_2=[0,255], color_3=[0,255], font_size = [0, 1], opacity=[0, 1], x_pos=[0, 0.5], y_pos=[0, 0.5], name = 'RandomOverlayText'):
        self.num = num
        self.text = text
        self.color_1 = color_1
        self.color_2 = color_2
        self.color_3 = color_3
        self.opacity = opacity
        self.font_size = font_size
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.name = name

    def __call__(self, x):
        for i in range(self.num):
            text = random.choices(range(100), k = random.randint(self.text[0],self.text[1]))
            color = [random.randint(self.color_1[0],self.color_1[1]),
                     random.randint(self.color_2[0],self.color_2[1]),
                     random.randint(self.color_3[0],self.color_3[1])]
            opacity = random.uniform(self.opacity[0], self.opacity[1])
            font_size = random.uniform(self.font_size[0], self.font_size[1])
            x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
            y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
            x = OverlayText(text = text,
                            font_size = font_size,
                            opacity = opacity,
                            color = color,
                            x_pos = x_pos,
                            y_pos = y_pos)(x)
        return x

class GrayScale:
    def __init__(self, name = 'GrayScale'):
        self.name = name
        
    def __call__(self, x):
        x = Grayscale()(x)
        return x
    
class RandomMemeFormat:
    def __init__(self, text_len = [1, 10], path = '/gsdata/home/wangwh/DGICD_dgx/DGICD/data/fonts/', opacity = [0, 1], \
                text_colors_0 = [0, 255], text_colors_1 = [0, 255], text_colors_2 = [0, 255], \
                caption_height = [100, 300], \
                bg_colors_0 = [0, 255], bg_colors_1 = [0, 255], bg_colors_2 = [0, 255], name = 'RandomMemeFormat'):
        self.text_len = text_len
        self.path = path
        self.opacity = opacity
        self.text_colors_0 = text_colors_0
        self.text_colors_1 = text_colors_1
        self.text_colors_2 = text_colors_2
        self.caption_height = caption_height
        self.bg_colors_0 = bg_colors_0
        self.bg_colors_1 = bg_colors_1
        self.bg_colors_2 = bg_colors_2
        self.name = name
    
    def __call__(self, x):
        string = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        length = random.randint(self.text_len[0], self.text_len[1])
        text = ''.join(random.sample(string, length))
        tiff_path = self.path + random.choice(os.listdir(self.path))
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        text_color_0 = random.randint(self.text_colors_0[0], self.text_colors_0[1])
        text_color_1 = random.randint(self.text_colors_1[0], self.text_colors_1[1])
        text_color_2 = random.randint(self.text_colors_2[0], self.text_colors_2[1])
        height = random.randint(self.caption_height[0], self.caption_height[1])
        bg_color_0 = random.randint(self.bg_colors_0[0], self.bg_colors_0[1])
        bg_color_1 = random.randint(self.bg_colors_1[0], self.bg_colors_1[1])
        bg_color_2 = random.randint(self.bg_colors_2[0], self.bg_colors_2[1])
        x = MemeFormat(text = text,
                       font_file = tiff_path,
                       opacity = opacity,
                       text_color = (text_color_0, text_color_1, text_color_2),
                       caption_height= height,
                       meme_bg_color= (bg_color_0, bg_color_1, bg_color_2))(x)
        return x

class OverlayScreen:
    def __init__(self, name = 'OverlayScreen'):
        self.name = name
        
    def __call__(self, x):
        x = OverlayOntoScreenshot()(x)
        return x

class RandomOverlayStripes:
    def __init__(self, line_widths = [0, 1], \
                 line_color_0 = [0, 255], line_color_1 = [0, 255], line_color_2 = [0, 255], \
                 line_angles = [0, 360], line_densitys = [0, 1], line_opacitys = [0, 1], name = 'RandomOverlayStripes'):
        self.line_widths = line_widths
        self.line_color_0 = line_color_0
        self.line_color_1 = line_color_1
        self.line_color_2 = line_color_2
        self.line_angles = line_angles
        self.line_densitys = line_densitys
        self.line_opacitys = line_opacitys
        self.name = name
    
    def __call__(self, x):
        line_width = random.uniform(self.line_widths[0], self.line_widths[1])
        line_color = (random.randint(self.line_color_0[0], self.line_color_0[1]), \
                       random.randint(self.line_color_1[0], self.line_color_1[1]), \
                       random.randint(self.line_color_2[0], self.line_color_2[1]))
        line_angle = random.randint(self.line_angles[0], self.line_angles[1])
        line_density = random.uniform(self.line_densitys[0], self.line_densitys[1])
        line_opacity = random.uniform(self.line_opacitys[0], self.line_opacitys[1])
        
        x = OverlayStripes(line_width = line_width, \
                           line_color = line_color, \
                           line_angle = line_angle, \
                           line_density = line_density, \
                           line_opacity = line_opacity)(x)
        
        return x

class RandomAddNoise:
    def __init__(self, means = [0, 0.5], varrs = [0, 0.5], name = 'RandomAddNoise'):
        self.means = means
        self.varrs = varrs
        self.name = name
        
    def __call__(self, x):
        mean = random.uniform(self.means[0], self.means[1])
        var = random.uniform(self.varrs[0], self.varrs[1])
        x = RandomNoise(mean = mean, var = var)(x)
        return x

class RandomSharpen:
    def __init__(self, factors = [1, 10], name = 'RandomSharpen'):
        self.factors = factors
        self.name = name
    
    def __call__(self, x):
        factor = random.uniform(self.factors[0], self.factors[1])
        x = Sharpen(factor = factor)(x)
        return x

class RandomSkew:
    def __init__(self, skew_factors = [-2, 2], name = 'RandomSkew'):
        self.skew_factors = skew_factors
        self.name = name
    
    def __call__(self, x):
        skew_factor = random.uniform(self.skew_factors[0], self.skew_factors[1])
        x = Skew(skew_factor = skew_factor)(x)
        return x

class VertFlip:
    def __init__(self, name = 'VertFlip'):
        self.name = name
        
    def __call__(self, x):
        return VFlip()(x)

path_1 = '/gsdata/home/wangwh/DGICD_dgx/DGICD/data/training_images/'
path_2 = '/gsdata/home/wangwh/DGICD_dgx/DGICD/data/train_v1_name_same/train_v1_name_same/'

names = sorted(os.listdir(path_1))
os.makedirs(path_2, exist_ok=True)

parser = argparse.ArgumentParser()
def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)
group = parser.add_argument_group('The range of images')
aa('--num', default=0, type=int, help="The begin number ")
args = parser.parse_args()

num = args.num
begin = num * 4000
end = (num+1) * 4000

all_names = []
for i in range(begin, end):
    if(i%10==0):
        print('processing...',i)
        image = Image.open(path_1 + names[i])
        name = str(i//10)+'_0.jpg'
        image.resize((256,256)).save(path_2 + name, quality=100)
        for j in range(1,20):
            transform_q = transforms.Compose(
                [ToRGB(), RandomResizeCrop()] + 
                random.sample([
                    RandomBlur(),
                    RandomSaturation(),
                    RandomOverlayText(),
                    GrayScale(),
                    RandomMemeFormat(),
                    RandomOverlayStripes(),
                    RandomAddNoise(),
                    RandomSharpen(),
                    RandomSkew(),
                    VertFlip(),
                    OverlayScreen(),
                ], 3) + 
                [transforms.Resize((256,256)), ToRGB()]
            )
            try:
                image_q = transform_q(image)
            except:
                transform_q = transforms.Compose(
                    [ToRGB(), RandomResizeCrop()] + 
                    random.sample([
                        RandomBlur(),
                        RandomSaturation(),
                        RandomOverlayText(),
                        GrayScale(),
                        RandomMemeFormat(),
                        RandomOverlayStripes(),
                        RandomAddNoise(),
                        RandomSharpen(),
                        RandomSkew(),
                        VertFlip(),
                        OverlayScreen(),
                    ], 3) + 
                    [transforms.Resize((256,256)), ToRGB()]
                )
                image_q = transform_q(image)
            
            name = str(i//10)+'_'+ str(j) +'.jpg'
            image_q.save(path_2 + name, quality=100)
            names_t = [name] + [t.name for t in transform_q.transforms[2:5]]
            all_names.append(names_t)

df = pd.DataFrame(all_names)
df.columns = ['name', 'pattern_1', 'pattern_2', 'pattern_3']
df.to_csv('names_same/' + str(num) + '.csv', index = False)
