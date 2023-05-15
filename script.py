#!/usr/bin/env python3

def get_prologue(barabasi, caption):
    return ['\\begin{table}[!htbp]', '\\vspace{-5pt}', '\\centering', '\\caption{' + caption + '}', '\\label{tab:mini_results}', '{\small', '\\begin{tabular}{llll}' if barabasi else '\\begin{tabular}{llllll}', '\\toprule', 'Sampler & $m = 2$ & $m = 3$ & $m = 4$ \\\\' if barabasi else 'Sampler & $\\alpha = 0.1$ & $\\alpha = 0.25$ & $\\alpha = 0.5$ & $\\alpha = 0.75$ & $\\alpha = 0.9$ \\\\', '\\midrule']

def main():
    epilogue = ['\\end{tabular}', '}', '\\vspace{-10pt}', '\\end{table}']

    i = 0
    with open('results-barabasi.csv') as f:
        lines = list(map(lambda x: list(filter(lambda x: len(x) != 0, x.split(','))), [line.strip() for line in f]))
        for line in lines:
            if 'alpha=0.1' in line or len(line) == 0:
                continue
            if line[0].isupper() or 'n = ' in line[0]:
                print(line[0] + ' & ', end='')
                for i in range(1, len(line)):
                    word = line[i]
                    if i % 2 == 1:
                        print(f'{float(word):0.4f} ± ', end='')
                    elif i + 1 != len(line):
                        print(f'{float(word):0.4f} & ', end='')
                    else:
                        print(f'{float(word):0.4f} \\\\')
            else:
                if i != 0:
                    for w in epilogue:
                        print(w)
                    print()
                prologue = []
                if 'Barabasi' in line[0]:
                    prologue = get_prologue(True, line[0])
                else:
                    prologue = get_prologue(False, line[0])
                for w in prologue:
                    print(w)
                i = i + 1
    for line in epilogue:
        print(line)
    

if __name__ == '__main__':
    main()