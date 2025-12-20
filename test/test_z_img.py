import torch
from diffusers import ZImagePipeline

model_path = r"F:\T2ITrainer_pulic\T2ITrainer\z_img_models\Z-Image-Turbo"
pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.to("cuda")
# pipe.enable_sequential_cpu_offload()
prompt = "一幅为名为“造相「Z-IMAGE-TURBO」”的项目设计的创意海报。画面巧妙地将文字概念视觉化：一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。"
# prompt = "自拍照，没有明确的主体或构图感，就是随手一拍的快照。照片略带运动模糊，阳光或室内打光不均匀导致的轻微曝光过度，xhs美女，前置摄像头自拍照，棕色长发，美女，表情自然松弛，水光肌，含情杏眼，前胸饱满，纯欲风，葫芦形S身材，身材火辣傲人，上围挺拔饱满，沙漏型身材，鬓角发丝凌乱自然垂落，绝美的锁骨，性感大胆，傲人身材，回头，依靠，一只手伸向镜头，- 服饰：上身是一件米白色的系带衬衫，领口的蝴蝶结设计很精致；下装是一条黑色波点短裙，凸显身材比例。- 鞋履：浅黑色的尖头高跟鞋，鞋头有蝴蝶结装饰，脚踝处还搭配了珍珠脚链，增添了精致感。长安汽车展厅 - 配饰：手腕上戴的手链，细节处很有心思。"
# Depending on the variant being used, the pipeline call will slightly vary.
# Refer to the pipeline documentation for more details.
image = pipe(
    prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("test.png")