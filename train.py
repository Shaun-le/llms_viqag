from lmqg import Trainer

trainer = Trainer(
    checkpoint_dir='./checkpoint',
    dataset_path='shnl/test',
    model='vinai/bartpho-syllable-base',
    use_auth_token = False,
    epoch=10,
    batch=8,
    lr=1e-04,
    prefix_types = ['qg','ae'],
    input_types = ['paragraph_answer', 'paragraph_sentence'],
    output_types = ['question', 'answer'],
    max_length = 512,
    max_length_output = 256
)

trainer.train()