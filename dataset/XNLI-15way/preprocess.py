import pandas as pd


def extract_sent():
    path = "./xnli.15way.orig.tsv"
    df = pd.read_csv(path, sep='\t', header=0)

    save_path = "./en_es_pair.csv"
    df[['en', 'es']].to_csv(save_path)

def construct_pair():
    readpath = "./en_es_pair.csv"
    writepath = "./en_es_siam.csv"
    df = pd.read_csv(readpath, header=0, index_col=0)
    new_df = pd.DataFrame(columns=('en', 'es', 'label'))
    for i in range(df.shape[0]):
        new_df.loc[2*i, 'en'] = df.loc[i, 'en']
        new_df.loc[2*i, 'es'] = df.loc[i, 'es']
        new_df.loc[2*i, 'label'] = 1
        new_df.loc[2*i + 1, 'en'] = df.loc[i, 'en']
        new_df.loc[2*i + 1, 'es'] = df.loc[(i + 1000) % df.shape[0], 'es']
        new_df.loc[2*i + 1, 'label'] = 0
    new_df.to_csv(writepath)

if __name__ == "__main__":
    construct_pair()
