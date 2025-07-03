1. 拉取TrajectoryCrafter
2. demo：mytest，其中的mask.mp4、render.mp4等文件是通过myinference.py产生
3. 训练代码：mytrain.py
4. 推理代码：myinference.py，是通过内外参矩阵直接控制

[Optional]为了方便您阅读代码：

1.核心数据流：Inference.py(pvd = TrajCrafter(opts))->demo.py(self.pipeline)->pipeline_trajectorycrafter.py(self.transformer)->crosstransformer3d.py(forward)


2.在哪里完成concat的：crosstransformer3d.py 694

hidden_states = torch.concat([hidden_states, inpaint_latents], 2)


3.inpaint的构成代码：inpaint_latents = torch.cat([mask_input, masked_video_latents_input], dim=1)看起来像mask，为什么表示的是镂空图？

masked_video = ( init_video * (mask_condition_tile < 0.5)\+ torch.ones_like(init_video) \* (mask_condition_tile > 0.5)\* -1)，这个只是表示apply_mask，如此命名而已


4.render.mp4文件在哪里保存：/home/tione/notebook/TrajectoryCrafter/demo.py 223
