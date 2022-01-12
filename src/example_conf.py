import configargparse

def main():
    print('------Nuevamente las opciones--------')
    print(options)
    print('GENOME', options.genome)
    print('Conf file', options.my_config)

if __name__ == '__main__':
    p = configargparse.ArgParser(default_config_files=['/etc/app/conf.d/*.conf', '~/.my_settings'])
    p.add('-c', '--my_config', required=True, is_config_file=True, help='config file path')
    p.add('-e', '--exp', help='Exp')

    p.add('--genome', required=True, help='path to genome file')  # this option can be set in a config file because it starts with '--'
    p.add('-v', help='verbose', action='store_true')
    p.add('-d', '--dbsnp', help='known variants .vcf', env_var='DBSNP_PATH')  # this option can be set in a config file because it starts with '--'
    p.add('vcf', nargs='+', help='variant file(s)')
    p.add("--la_leche")

    p.add_argument(
        '--train_batch_size',
        help='Batch size for training steps',
        type=int,
        default=5
    )

    p.add_argument(
        "--num_channels",
        help="Num Channels at the model [1,2,4,8,16]",
        default=16,
        choices=[1, 2, 4, 8, 16],
        type=int
    )

    options = p.parse_args()

    print(options)
    print("----------")
    print(p.format_help())
    print("----------")
    print(p.format_values())    # useful for logging where different settings came from
    print("----------")
    print(p.print_values())
    p.write_config_file(options, ['/home/pmacias/ex_file2.txt'])
    main()