from pathlib import Path

from tqdm import tqdm

from .args import ArgumentParserBuilder, OptionEnum
from vizbert.utils import quick_nlp


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(OptionEnum.DATA_FOLDER,
                 OptionEnum.OUTPUT_FOLDER)
    args = apb.parser.parse_args()

    nlp = quick_nlp(name='tokenize')
    df = args.data_folder  # type: Path
    of = args.output_folder  # type: Path
    try:
        of.mkdir()
    except:
        pass
    for filename in ('train.tsv', 'dev.tsv', 'test.tsv'):
        filename1 = df / filename
        filename2 = of / filename
        with open(filename1) as f1, open(filename2, 'w') as f2:
            for line in tqdm(iter(f1.readline, '')):
                line = line.strip()
                label, doc = line.split('\t')
                nlp_doc = nlp(doc)
                if len(nlp_doc.sentences) == 1:
                    f2.write(f'{line}\n')
                    continue
                doc = ' '.join([s.text for s in nlp_doc.sentences[:-1]])
                f2.write(f'{label}\t{doc}\n')


if __name__ == '__main__':
    main()
