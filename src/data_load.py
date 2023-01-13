import pickle
import urllib.error
import urllib.request

import pandas as pd



def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
        flag = True
    except:
        flag = False
        print(dst_path)
    return flag

def get_fig_from_url(urls):
    fig_paths = []
    rm_idx = []
    cnt = 0
    for i, url in enumerate(urls):
        path = "../data/this_is_gallery/fig/{}.jpg".format(cnt)
        flag = download_file(url, path)

        if flag:
            fig_paths.append(path)
            cnt += 1
        else:
            rm_idx.append(i)
    
    print(len(fig_paths))
    
    return fig_paths, rm_idx


def label_normalize(labels, rm_idx):
    labels = labels.drop(index=rm_idx)
    print(len(labels))

    mean_labels = labels.mean()
    std_labels = labels.std()

    rt_labels = (labels - mean_labels) / std_labels

    return rt_labels, [mean_labels, std_labels]


def preprocessing_data(df):
    fig_urls = df['src']
    labels = df['price']
    labels = labels.astype(int)

    fig_paths, rm_idx = get_fig_from_url(fig_urls)
    labels, normalize_para = label_normalize(labels, rm_idx)

    paths = pd.DataFrame(fig_paths)

    paths = paths.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    rt_df = pd.concat([paths, labels], axis=1, ignore_index=True)
    rt_df.columns = ['paths','labels']

    with open("../data/this_is_gallery/normalize_para.pickle", "wb") as f:
        pickle.dump(normalize_para, f)

    return rt_df

if __name__ == '__main__':
    with open('../data/this_is_gallery/df.pickle', 'rb') as f:
        df = pickle.load(f)

    rt_df = preprocessing_data(df)
    print(rt_df)
    
    with open('../data/this_is_gallery/preprocess_df.pickle', 'wb') as f:
        pickle.dump(rt_df, f)