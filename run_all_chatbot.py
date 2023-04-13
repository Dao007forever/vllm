import os


def run_cmd(cmd):
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        exit()


if __name__ == "__main__":
    rates = [1.0, 0.8, 0.6, 0.4, 0.2]
    duration = 1000
    estimator = "oracle"
    for rate in rates:
        cmd = f"python benchmark/benchmark_chatbot.py --len-estimator {estimator} --dataset sharegpt_clean_lang_10k_opt_tokenized.pkl --model facebook/opt-13b --request-rate {rate} --duration {duration} --n1 1.0 --use-dummy"
        run_cmd(cmd)
