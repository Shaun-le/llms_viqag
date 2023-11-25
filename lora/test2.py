import torch
from peft import PeftModel
import transformers
import textwrap
import fire
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, DataCollatorForSeq2Seq
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
import os
import json
import os.path as osp
from typing import Union
from datasets import load_dataset
from tqdm import tqdm


def test(
        base_model: str = 'manhtt-079/llama-2-7b',
        lora_weight: str = '/home/int2-user/llms_viqag/lora/weight',
        check_point: str = '',
        test_path: str = '',
        inst: str = 'Viết các cặp câu hỏi và câu trả lời dựa trên đoạn văn bản sau.',
        inp: str = '1.000 ngày đầu đời được tính từ lúc bé còn trong bụng mẹ khoảng 270 ngày và 730 ngày tiếp theo khi chào đời. Nhiều khía cạnh của sức khỏe lâu dài được xây dựng trong 1.000 ngày, trong đó có hệ vi sinh vật đường ruột. Để hệ vi sinh vật đường ruột được củng cố và tăng cường, bé cần bú sữa mẹ càng sớm càng tốt ngay sau khi sinh. Sữa mẹ có nhiều lợi khuẩn như Bifidus, B. lactis aerogenes, các loại men beta lactoza... tốt cho đường ruột cũng như ức chế vi khuẩn Ecoli. Đạm Whey, giàu Alpha-Lactalbumin giúp con trẻ dễ tiêu hóa, tăng cường sức đề kháng cũng chiếm đến 60% trong sữa mẹ. Trẻ nhỏ cần ăn uống theo nhu cầu, phù hợp trong từng giai đoạn. Khi trẻ bắt đầu tiếp xúc với nguồn dinh dưỡng ngoài, mẹ cần cung cấp chế độ ăn uống đầy đủ 4 nhóm chất cần thiết gồm chất bột đường, đạm, béo, vitamin và khoáng chất. Các loại rau xanh, ngũ cốc, trái cây... giàu chất xơ có lợi cho đường tiêu hóa. Các loại hoa quả mềm, nhiều nước như chuối, đu đủ, dưa hấu, cam, bơ, bưởi... giúp hệ tiêu hóa dần làm quen và thích ứng. Củ, quả cần luộc thật mềm, rau chỉ nên chọn phần non nhất và phần lá, ưu tiên các loại đạm dễ tiêu có nhiều trong cá, tôm, cua... Ông Gernot Stadlmann, Giám đốc khu vực Châu Á - Thái Bình Dương, Tập đoàn dinh dưỡng hàng đầu Châu Âu Chr. Hansen Đan Mạch cũng cho biết, trong giai đoạn 1.000 ngày đầu đời này, nhóm lợi khuẩn ở trẻ nhỏ còn hạn chế về mặt chủng loại và số lượng nên lượng đường Lactose và đạm sữa cần thiết cho sự phát triển của trẻ không được tiêu hóa hết. Chúng được chuyển thẳng xuống ruột già, nơi có chứa nhiều vi khuẩn sống dẫn đến chứng khó tiêu, rối loạn tiêu hóa, nôn trớ... Bổ sung men vi sinh Probiotics là một trong những cách giúp tối ưu hóa sinh vật có lợi trong đường ruột của trẻ ngay từ giai đoạn đầu đời. Các vi sinh vật có lợi sẽ tạo lớp hàng rào bảo vệ ở đường ruột, ngăn cản sự xâm nhập và tấn công của các hại khuẩn. Vi sinh vật có lợi còn sản xuất một số enzyme (men tiêu hóa), vitamin nội sinh tăng cường tiêu hóa và hấp thụ thức ăn. Trên thế giới, hai chủng loại men vi sinh Probiotics được nghiên cứu và ứng dụng nhiều là Lactobacillus rhamnosus GG (LGG) và Bifidobacterium (BB-12). Các nghiên cứu thử nghiệm lâm sàng có đối chứng của Tập đoàn Chr. Hansen về lợi điểm của LGG và BB-12 trên trẻ sơ sinh, trẻ nhỏ cho thấy kết quả tích cực. Hai chủng men vi sinh này giúp giảm tỷ lệ tiêu chảy (cấp tính hoặc liên quan đến kháng sinh) ở trẻ, hỗ trợ co bóp dạ dày tiêu hóa thức ăn nhanh, hấp thụ chất dinh dưỡng tốt hơn. Tỷ lệ quấy khóc ảnh hưởng trực tiếp đến tiêu hóa và giấc ngủ của bé cũng giảm. Không chỉ góp phần cho hệ tiêu hóa khỏe mạnh, LGG và BB-12 còn giúp tăng cường hệ miễn dịch, giảm tỷ lệ nhiễm trùng đường hô hấp, viêm da dị ứng, nhiễm trùng tai...'
):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(model, lora_weight, torch_dtype=torch.float16)
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model = model.eval()
    model = torch.compile(model)

    def generate_prompt(data_point):
        return f"""Dưới đây là một Instruction mô tả một nhiệm vụ, kèm theo một Input cung cấp ngữ cảnh. Viết một Response hoàn thành yêu cầu một cách thích hợp.

        ### Instruction:
        {data_point["instruction"]}

        ### Input:{
        data_point["input"]}"""

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenizer(full_prompt, truncation=True,padding=False,return_tensors=None)
        return tokenized_full_prompt

    def create_prompt(instruction: str, input: str) -> str:
        return PROMPT_TEMPLATE.replace("[INSTRUCTION]", instruction).replace("[INPUT]", input)

    def generate_response(prompt: str, model: PeftModel) -> GreedySearchDecoderOnlyOutput:
        encoding = tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"].to('cuda')

        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4
        )
        with torch.inference_mode():
            return model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
            )

    test = load_dataset("json", data_files=test_path)
    test_data = (
        test["train"].shuffle().map(generate_and_tokenize_prompt)
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    )

    dataloader = torch.utils.data.DataLoader(test_data, collate_fn=data_collator, batch_size=128)

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
    )

    predictions = []

    for _, batch in enumerate(tqdm(dataloader)):
        outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
        )

        with tokenizer.as_target_tokenizer():
            outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in
                       outputs.sequences]

        predictions.extend(outputs.split("### Response:")[1].strip())

    with open(os.path.join(check_point, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({'predictions': predictions}, f, ensure_ascii=False, indent=2)

    def format_response(response: GreedySearchDecoderOnlyOutput) -> str:
        decoded_output = tokenizer.decode(response.sequences[0])
        # print(decoded_output)
        response = decoded_output.split("### Response:")[1].strip()
        return "\n".join(textwrap.wrap(response))

    def ask_alpaca(inst: str, inp: str, model: PeftModel = model) -> str:
        prompt = create_prompt(inst, inp)
        response = generate_response(prompt, model)
        print(format_response(response))

    print(ask_alpaca(inst,inp))

if __name__ == "__main__":
    fire.Fire(test)