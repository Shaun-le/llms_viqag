import torch
from peft import PeftModel
import transformers
import textwrap
import fire
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ask(
        base_model: str = 'manhtt-079/llama-2-7b',
        lora: str = '',
        inst: str = 'Tạo ra một vài cặp câu hỏi và câu trả lời tương ứng với đoạn văn bản sau.',
        inp: str = "\"9 tháng 10 ngày mang thai là thời gian ý nghĩa nhất trong cuộc đời làm mẹ\", Thủy Anh - giám đốc điều hành một công ty truyền thông ở TP HCM tâm sự. Khi mang thai con trai đầu lòng, Thủy Anh rất tự ti vì tăng gần 15 kg. Da bị rạn, cơ thể không săn chắc. Do vậy, khi mang thai lần hai cô quyết tâm giữ dáng để \"tăng cho con, thon cho mẹ\". Nhiều bà bầu quan niệm \"mẹ tăng cân nhiều thì con mới khỏe\". Theo bác sĩ chuyên khoa II Nguyễn Thị Tân Sinh, Bệnh viện Bạch Mai, mẹ tăng cân quá mức là nguy cơ dẫn đến tiểu đường, tiền sản giật và nguy hiểm cho thai nhi. Bởi vậy lần mang bầu thứ hai Thủy Anh luôn kiểm soát cân nặng lý tưởng cả thai kỳ 11-13 kg. Cô nhờ bác sĩ xây dựng chế độ dinh dưỡng phù hợp. 3 tháng đầu chỉ nên tăng 2-3 kg, Thủy Anh ăn thực phẩm nhiều DHA, axit folic, canxi có trong cá, rau củ quả, trứng... hạn chế chất béo. Đây là giai đoạn mẹ bầu thường xuyên bị nghén nên phải chia nhỏ bữa ăn, không ăn quá no, hạn chế ăn nhiều tinh bột. Đến 3 tháng giữa của thai kỳ, em bé phát triển nhanh hơn. Mẹ bầu uống nhiều sữa, bổ sung can xi, ăn nhiều hải sản và rau xanh. Thủy Anh chọn uống sữa không đường để giảm tối đa lượng đường nạp vào cơ thể. Giai đoạn này bà bầu cần tăng 4-5 kg để đảm bảo sức khỏe cho bé. Ba tháng cuối thai kỳ là giai đoạn em bé cần dinh dưỡng để phát triển hoàn thiện. Lúc này bà bầu ăn uống giàu đạm, protein, chất béo và các chế phẩm từ sữa như sữa chua, phô mai, sữa tươi không đường. Khi mang bầu, cơ thể cần nhiều năng lượng hơn bình thường nên Thủy Anh rất thèm ăn và ăn nhiều gấp 3 lần bình thường. \"Nhiều lần vì lỡ ăn nhiều nên bữa ăn sau phải giảm bớt, chỉ ăn rau xanh, hoa quả thôi\", cô chia sẻ. Vì vậy, trong nhà cô luôn có sẵn một chiếc cân để kiểm soát cân nặng mỗi ngày. Mỗi ngày Thủy Anh tập những bài tập thể dục cho bà bầu để rèn luyện sức khỏe, giảm đau lưng và hông. Cô khuyên mẹ bầu nếu có thời gian nên tập yoga, đi bộ hay bơi để luôn thoải mái và có năng lượng bắt đầu một ngày mới. Trong thai kỳ, mẹ bầu thường xuất hiện vết rạn ở bụng, tay, chân. Để khắc phục, Thủy Anh dùng dầu dừa để xoa các vết rạn trước khi ngủ. Dầu dừa là tinh chất tự nhiên và an toàn để làm khi mang thai. Những ngày mang bầu cũng là thời gian hạnh phúc của Thủy Anh khi có sự hỗ trợ của chồng. Cô rất thích được cùng chồng đi dạo, đọc sách và san sẻ việc nhà. Sự quan tâm của chồng là liều thuốc tinh thần giúp bà bầu chống lại stress. \"Anh ấy nói: 'Em chỉ việc khỏe mạnh sinh con thôi, còn lại để anh lo'\", Thủy Anh chia sẻ sau khi sinh con thứ hai. \"Tăng cân hợp lý khi mang bầu giúp bé khỏe mạnh và mẹ bầu nhanh chóng lấy lại vóc dáng ban đầu\", Thủy Anh đúc kết kinh nghiệm. Hiện tại, cô là mẹ của hai bé Đăng Khang và Đăng Anh."
):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, lora, torch_dtype=torch.float16)
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model = model.eval()
    model = torch.compile(model)

    PROMPT_TEMPLATE = f"""
    Dưới đây là một Instruction mô tả một nhiệm vụ, kèm theo một Input cung cấp ngữ cảnh. Viết một Response hoàn thành yêu cầu một cách thích hợp.

    ### Instruction:
    [INSTRUCTION]
    
    ### Input:
    [INPUT]

    ### Response:
    """

    def create_prompt(instruction: str, input: str) -> str:
        return PROMPT_TEMPLATE.replace("[INSTRUCTION]", instruction).replace("[INPUT]", input)

    def generate_response(prompt: str, model: PeftModel) -> GreedySearchDecoderOnlyOutput:
        encoding = tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"].to(DEVICE)

        generation_config = GenerationConfig(
            temperature=0.05,
            top_p=0.75,
            top_k=40,
            #num_beams=4,
        )
        #with torch.inference_mode():
            #return (
        return model.generate(
                input_ids=input_ids,
                #generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                temperature=0.2
        )

    def format_response(response: GreedySearchDecoderOnlyOutput) -> str:
        decoded_output = tokenizer.decode(response.sequences[0])
        response = decoded_output.split("### Response:")[1].strip()
        return "\n".join(textwrap.wrap(response))

    def ask_alpaca(inst: str, inp:str, model: PeftModel = model) -> str:
        prompt = create_prompt(inst,inp)
        response = generate_response(prompt, model)
        print(format_response(response))

    print(ask_alpaca(inst,inp))

if __name__ == "__main__":
    fire.Fire(ask)