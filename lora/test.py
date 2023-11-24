import torch
from peft import PeftModel
import transformers
import textwrap
import fire
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput

import json
import os.path as osp
from typing import Union
class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
            print(res)
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
def test(
        base_model: str = 'manhtt-079/llama-2-7b',
        lora_weight: str = '/home/int2-user/llms_viqag/lora/weight',
        inst: str = 'Viết các cặp câu hỏi và câu trả lời dựa trên đoạn văn bản sau.',
        inp: str = '1.000 ngày đầu đời được tính từ lúc bé còn trong bụng mẹ khoảng 270 ngày và 730 ngày tiếp theo khi chào đời. Nhiều khía cạnh của sức khỏe lâu dài được xây dựng trong 1.000 ngày, trong đó có hệ vi sinh vật đường ruột. Để hệ vi sinh vật đường ruột được củng cố và tăng cường, bé cần bú sữa mẹ càng sớm càng tốt ngay sau khi sinh. Sữa mẹ có nhiều lợi khuẩn như Bifidus, B. lactis aerogenes, các loại men beta lactoza... tốt cho đường ruột cũng như ức chế vi khuẩn Ecoli. Đạm Whey, giàu Alpha-Lactalbumin giúp con trẻ dễ tiêu hóa, tăng cường sức đề kháng cũng chiếm đến 60% trong sữa mẹ. Trẻ nhỏ cần ăn uống theo nhu cầu, phù hợp trong từng giai đoạn. Khi trẻ bắt đầu tiếp xúc với nguồn dinh dưỡng ngoài, mẹ cần cung cấp chế độ ăn uống đầy đủ 4 nhóm chất cần thiết gồm chất bột đường, đạm, béo, vitamin và khoáng chất. Các loại rau xanh, ngũ cốc, trái cây... giàu chất xơ có lợi cho đường tiêu hóa. Các loại hoa quả mềm, nhiều nước như chuối, đu đủ, dưa hấu, cam, bơ, bưởi... giúp hệ tiêu hóa dần làm quen và thích ứng. Củ, quả cần luộc thật mềm, rau chỉ nên chọn phần non nhất và phần lá, ưu tiên các loại đạm dễ tiêu có nhiều trong cá, tôm, cua... Ông Gernot Stadlmann, Giám đốc khu vực Châu Á - Thái Bình Dương, Tập đoàn dinh dưỡng hàng đầu Châu Âu Chr. Hansen Đan Mạch cũng cho biết, trong giai đoạn 1.000 ngày đầu đời này, nhóm lợi khuẩn ở trẻ nhỏ còn hạn chế về mặt chủng loại và số lượng nên lượng đường Lactose và đạm sữa cần thiết cho sự phát triển của trẻ không được tiêu hóa hết. Chúng được chuyển thẳng xuống ruột già, nơi có chứa nhiều vi khuẩn sống dẫn đến chứng khó tiêu, rối loạn tiêu hóa, nôn trớ... Bổ sung men vi sinh Probiotics là một trong những cách giúp tối ưu hóa sinh vật có lợi trong đường ruột của trẻ ngay từ giai đoạn đầu đời. Các vi sinh vật có lợi sẽ tạo lớp hàng rào bảo vệ ở đường ruột, ngăn cản sự xâm nhập và tấn công của các hại khuẩn. Vi sinh vật có lợi còn sản xuất một số enzyme (men tiêu hóa), vitamin nội sinh tăng cường tiêu hóa và hấp thụ thức ăn. Trên thế giới, hai chủng loại men vi sinh Probiotics được nghiên cứu và ứng dụng nhiều là Lactobacillus rhamnosus GG (LGG) và Bifidobacterium (BB-12). Các nghiên cứu thử nghiệm lâm sàng có đối chứng của Tập đoàn Chr. Hansen về lợi điểm của LGG và BB-12 trên trẻ sơ sinh, trẻ nhỏ cho thấy kết quả tích cực. Hai chủng men vi sinh này giúp giảm tỷ lệ tiêu chảy (cấp tính hoặc liên quan đến kháng sinh) ở trẻ, hỗ trợ co bóp dạ dày tiêu hóa thức ăn nhanh, hấp thụ chất dinh dưỡng tốt hơn. Tỷ lệ quấy khóc ảnh hưởng trực tiếp đến tiêu hóa và giấc ngủ của bé cũng giảm. Không chỉ góp phần cho hệ tiêu hóa khỏe mạnh, LGG và BB-12 còn giúp tăng cường hệ miễn dịch, giảm tỷ lệ nhiễm trùng đường hô hấp, viêm da dị ứng, nhiễm trùng tai...'
):
    prompter = Prompter('me')
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto")
    model = PeftModel.from_pretrained(
        model,
        lora_weight,
        torch_dtype=torch.float16)

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model = model.eval()
    model = torch.compile(model)

    prompt = prompter.generate_prompt(inst, inp)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(prompter.get_response(output))

if __name__ == "__main__":
    fire.Fire(test)