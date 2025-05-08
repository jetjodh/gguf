__version__ = '1.5.4'
def __init__():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version', version=
        '%(prog)s ' + __version__)
    subparsers = parser.add_subparsers(title='subcommands', dest=
        'subcommand', help='choose a subcommand:')
    subparsers.add_parser('cpp', help='[cpp] connector cpp')
    subparsers.add_parser('gpp', help='[gpp] connector gpp')
    subparsers.add_parser('v', help='[v] vision connector')
    subparsers.add_parser('g', help='[g] cli connector g')
    subparsers.add_parser('c', help='[c] gui connector c')
    subparsers.add_parser('m', help='[m] menu')
    subparsers.add_parser('o', help='[o] org web mirror')
    subparsers.add_parser('i', help='[i] i/o web mirror')
    subparsers.add_parser('w', help='[w] page/container')
    subparsers.add_parser('y', help='[y] download comfy')
    subparsers.add_parser('n', help='[n] clone node')
    subparsers.add_parser('u', help='[u] get cutter')
    subparsers.add_parser('p', help='[p] take pack')
    subparsers.add_parser('p1', help='[p1] take framepack')
    subparsers.add_parser('p2', help='[p2] take packpack')
    subparsers.add_parser('r', help='[r] metadata reader')
    subparsers.add_parser('r2', help='[r2] metadata fast reader')
    subparsers.add_parser('r3', help='[r3] tensor reader')
    subparsers.add_parser('q', help='[q] tensor quantizor')
    subparsers.add_parser('q2', help='[q2] tensor quantizor (upscale)')
    subparsers.add_parser('d', help='[d] divider (safetensors)')
    subparsers.add_parser('d2', help='[d2] divider (gguf)')
    subparsers.add_parser('ma', help='[ma] merger (safetensors)')
    subparsers.add_parser('m2', help='[m2] merger (gguf)')
    subparsers.add_parser('s', help='[s] splitter (checkpoint)')
    subparsers.add_parser('t', help='[t] tensor convertor')
    subparsers.add_parser('t0', help='[t0] tensor convertor (zero)')
    subparsers.add_parser('t1', help='[t1] tensor convertor (alpha)')
    subparsers.add_parser('t2', help='[t2] tensor convertor (beta)')
    subparsers.add_parser('t3', help='[t3] tensor convertor (gamma)')
    subparsers.add_parser('t4', help='[t4] tensor convertor (delta)')
    subparsers.add_parser('t5', help='[t5] tensor convertor (epsilon)')
    subparsers.add_parser('t6', help='[t6] tensor convertor (zeta)')
    subparsers.add_parser('t7', help='[t7] tensor convertor (eta)')
    subparsers.add_parser('t8', help='[t8] tensor convertor (theta)')
    subparsers.add_parser('t9', help='[t9] tensor convertor (iota)')
    subparsers.add_parser('d5', help='[d5] dimension 5 fixer (t8x)')
    subparsers.add_parser('pp', help='[pp] pdf analyzor pp')
    subparsers.add_parser('cp', help='[cp] pdf analyzor cp')
    subparsers.add_parser('ps', help='[ps] wav recognizor ps')
    subparsers.add_parser('cs', help='[cs] wav recognizor cs')
    subparsers.add_parser('cg', help='[cg] wav recognizor cg (online)')
    subparsers.add_parser('pg', help='[pg] wav recognizor pg (online)')
    subparsers.add_parser('vg', help='[vg] video generator (t2v)')
    subparsers.add_parser('v1', help='[v1] video generator (i2v)')
    subparsers.add_parser('v2', help='[v2] video 2 (t2v)')
    args = parser.parse_args()
    if args.subcommand == 'm':
        from gguf_connector import m
    if args.subcommand == 'n':
        from gguf_connector import n
    if args.subcommand == 'p':
        from gguf_connector import p
    if args.subcommand == 'p1':
        from gguf_connector import p1
    if args.subcommand == 'p2':
        from gguf_connector import p2
    elif args.subcommand == 'r':
        from gguf_connector import r
    elif args.subcommand == 'r2':
        from gguf_connector import r2
    elif args.subcommand == 'r3':
        from gguf_connector import r3
    elif args.subcommand == 's':
        from gguf_connector import s
    elif args.subcommand == 'i':
        from gguf_connector import i
    elif args.subcommand == 'o':
        from gguf_connector import o
    elif args.subcommand == 'u':
        from gguf_connector import u
    elif args.subcommand == 'v':
        from gguf_connector import v
    elif args.subcommand == 'vg':
        from gguf_connector import vg
    elif args.subcommand == 'v1':
        from gguf_connector import vg2
    elif args.subcommand == 'v2':
        from gguf_connector import v2
    elif args.subcommand == 'w':
        from gguf_connector import w
    elif args.subcommand == 'y':
        from gguf_connector import y
    elif args.subcommand == 't':
        from gguf_connector import t
    elif args.subcommand == 't0':
        from gguf_connector import t0
    elif args.subcommand == 't1':
        from gguf_connector import t1
    elif args.subcommand == 't2':
        from gguf_connector import t2
    elif args.subcommand == 't3':
        from gguf_connector import t3
    elif args.subcommand == 't4':
        from gguf_connector import t4
    elif args.subcommand == 't5':
        from gguf_connector import t5
    elif args.subcommand == 't6':
        from gguf_connector import t6
    elif args.subcommand == 't7':
        from gguf_connector import t7
    elif args.subcommand == 't8':
        from gguf_connector import t8
    elif args.subcommand == 't9':
        from gguf_connector import t9
    elif args.subcommand == 'd5':
        from gguf_connector import d5
    elif args.subcommand == 'q':
        from gguf_connector import q
    elif args.subcommand == 'q2':
        from gguf_connector import q2
    elif args.subcommand == 'd':
        from gguf_connector import d
    elif args.subcommand == 'd2':
        from gguf_connector import d2
    elif args.subcommand == 'm2':
        from gguf_connector import m2
    elif args.subcommand == 'ma':
        from gguf_connector import ma
    elif args.subcommand == 'cg':
        from gguf_connector import cg
    elif args.subcommand == 'pg':
        from gguf_connector import pg
    elif args.subcommand == 'cs':
        from gguf_connector import cs
    elif args.subcommand == 'ps':
        from gguf_connector import ps
    elif args.subcommand == 'cp':
        from gguf_connector import cp
    elif args.subcommand == 'pp':
        from gguf_connector import pp
    elif args.subcommand == 'c':
        from gguf_connector import c
    elif args.subcommand == 'cpp':
        from gguf_connector import cpp
    elif args.subcommand == 'g':
        from gguf_connector import g
    elif args.subcommand == 'gpp':
        from gguf_connector import gpp