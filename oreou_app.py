import numpy as np
import gradio as gr
from automatic_label_tag2text_demo import generate_fn
from automatic_label_simple_demo import segmentToimg
from pkg_resources import resource_filename
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)


with gr.Blocks() as demo:
    gr.Markdown(
        "# Grounded-SAM with Tag2Text for Automatic Labeling"
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image()
            seg_btn = gr.Button("语义分割图像")
            tag_btn = gr.Button("segment-anythine")
        with gr.Column(scale=2):
            # video_output = gr.Video(label='视频')
            seg_img = gr.Image(label='seg')
            describe = gr.Text(label='描述')
            tag = gr.Text(label='tag')
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="分割结果", show_label=True, elem_id="gallery"
            ).style(columns=[2], rows=[2], object_fit="contain", height="auto")

    # gr.Markdown("## 测试案例")
    # gr.Examples(
    #     examples=exmaple_list,
    #     inputs=[image_input, motion, gif_output],
    # )

    # text_button.click(text2img, inputs=text_input, outputs=image_input)
    # seg_btn.click(seg_fn, inputs=[image_input], outputs=seg_img)
    # tag_btn.click(tag_fn, inputs=[image_input], outputs=[describe, tag])
    seg_btn.click(generate_fn, inputs=[image_input], outputs=[seg_img, describe, tag], api_name="segmentOneAndGetTag")
    tag_btn.click(segmentToimg, inputs=[image_input], outputs=gallery, api_name="segmentAnything")

demo.launch(server_name="0.0.0.0", server_port=8888, show_api=True, share=True, app_kwargs={'docs_url': '/docs'})
