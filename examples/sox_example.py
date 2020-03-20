import sox  # https://github.com/rabitt/pysox
# sox can't read/write directly from/to memory ... works only on files
# for direct memory inp/out transormation use pysndfx

if __name__ == '__main__':
    infile = "../data/dobry_den.wav"
    outfile = "../data/dobry_den_transformed.wav"

    tfm = sox.Transformer()
#    tfm.input_format()
#    tfm.output_format()
    tfm.speed(0.9)
    tfm.build(infile, outfile)
