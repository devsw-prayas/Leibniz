#!/usr/bin/env python3
# striped_tables_gen.py — generates StripedTables.h
# Auto-generates AVX2 broadcast stripe tables for FP32 and FP64, Balanced and Approximate tiers.

import sys

TABLES = {
    'f32': {
        'balanced': {
            'Exp':              ['0x1.fff61cp-1f', '0x1.fffe06p-1f', '0x1.029248p-1f', '0x1.57636ep-3f'],
            'Expm1':            ['0x1.fffe06p-1f', '0x1.ffff58p-2f', '0x1.57636ep-3f', '0x1.56b3d8p-5f'],
            'Exp2':             ['0x1.fffdb8p-1f', '0x1.62e3ep-1f', '0x1.ee5a96p-3f', '0x1.c800d2p-5f'],
            'Exp10':            ['0x1.fee536p-1f', '0x1.269acp+1f', '0x1.65aaacp+1f', '0x1.0cd7acp+1f'],
            'Log1p':            ['-0x1.5ap-109f', '0x1.fd7234p-1f', '-0x1.fbb0fp-2f', '0x1.a35456p-2f', '-0x1.41a74ap-2f'],
            'Log2':             ['-0x1.0p-108f', '0x1.6f7cd8p+0f', '-0x1.6e38c6p-1f', '0x1.2e7b74p-1f', '-0x1.d00c42p-2f'],
            'Log10':            ['-0x1.76p-110f', '0x1.ba7fdp-2f', '-0x1.b8f996p-3f', '0x1.6c3994p-3f', '-0x1.17627ap-3f'],
            'Sin':              ['0x1.1p-1f', '-0x1.555556p-3f', '0x1.111112p-7f', '-0x1.a019f8p-13f', '0x1.71d732p-19f', '-0x1.a9508p-26f'],
            'Cos':              ['0x1.1p-1f', '-0x1.1p-2f', '0x1.555556p-5f', '-0x1.6c16b2p-10f', '0x1.a00e96p-16f', '-0x1.23c5c2p-22f'],
            'Tan':              ['0x1.fffff8p-1f', '0x1.555c1ap-2f', '0x1.10158ep-3f', '0x1.d44a86p-5f', '0x1.9bb3ccp-7f', '0x1.4db42ap-6f'],
            'Asin':             ['0x1.000002p+0f', '0x1.554dcp-3f', '0x1.350eaep-4f', '0x1.45ffp-5f', '0x1.9c87b6p-5f'],
            'Atan':             ['0x1.ffff52p-1f', '-0x1.54f2b6p-2f', '0x1.9011ep-3f', '-0x1.ed5b7ep-4f', '0x1.cf95ccp-5f', '-0x1.ae41ap-7f'],
            'Sinh':             ['0x1.1p-1f', '0x1.555556p-3f', '0x1.1110a6p-7f', '0x1.a17df2p-13f'],
            'Cosh':             ['0x1.1p-1f', '0x1.0p-1f', '0x1.555464p-5f', '0x1.6da754p-10f'],
            'Asinh':            ['0x1.fffffp-1f', '-0x1.5539f8p-3f', '0x1.2f8124p-4f', '-0x1.3ef194p-5f', '0x1.ce028cp-7f'],
            'Acosh':            ['0x1.6a09e6p+0f', '-0x1.e2b3dp-4f', '0x1.b1697p-6f', '-0x1.ecfc96p-8f', '0x1.cc0298p-10f'],
            'Atanh':            ['0x1.000006p+0f', '0x1.55452p-2f', '0x1.9d8f88p-3f', '0x1.f4c8aep-4f', '0x1.8fc48ap-3f'],
            'Erf':              ['0x1.20dd74p+0f', '-0x1.81268p-2f', '0x1.ce0ab2p-4f', '-0x1.b5b31p-6f', '0x1.428d76p-8f', '-0x1.281efap-11f'],
            'ErfcAsymptotic':   ['0x1.0p+0f', '-0x1.0p-1f', '0x1.8p-1f', '-0x1.ep+0f'],
            'SqrtSeed':         ['0x1.669044p-3f', '0x1.433fa4p+0f', '-0x1.ac8a38p-1f', '0x1.df42c2p-2f', '-0x1.42c35ap-4f'],
            'RsqrtSeed':        ['0x1.c6b9cep+0f', '-0x1.67cdcep+0f', '0x1.f3625ap-1f', '-0x1.7d02f2p-2f', '0x1.cd2876p-5f'],
            'CbrtSeed':         ['0x1.395abp-1f', '0x1.ba4fd4p-1f', '-0x1.23bd2cp-1f', '0x1.18e382p-2f', '-0x1.e2ee22p-5f'],
            'Lanczos':          ['0x1.1p-1f', '0x1.52429cp+9f', '-0x1.3ac8e8p+10f', '0x1.81a966p+9f', '-0x1.613ae6p+7f', '0x1.903c28p+3f', '-0x1.1bcb2ap-3f', '0x1.4b045ap-17f', '0x1.433a1cp-23f'],
        },
        'approximate': {
            'Exp':              ['0x1.0p+0f', '0x1.03dc68p+0f', '0x1.01ed76p-1f'],
            'Expm1':            ['0x1.0p+0f', '0x1.01ed76p-1f', '0x1.56dfc4p-3f'],
            'Exp2':             ['0x1.0p+0f', '0x1.6574fap-1f', '0x1.edc2ccp-3f'],
            'Exp10':            ['0x1.0p+0f', '0x1.3ec25cp+1f', '0x1.610b4p+1f'],
            'Log1p':            ['-0x1.0p-109f', '0x1.121528p+0f', '-0x1.1b7f38p-1f'],
            'Log2':             ['-0x1.4p-109f', '0x1.8b6aecp+0f', '-0x1.98ffe8p-1f'],
            'Log10':            ['-0x1.0p-110f', '0x1.dc215cp-2f', '-0x1.ec7c12p-3f'],
            'Sin':              ['0x1.1p-1f', '-0x1.55554p-3f', '0x1.11062ep-7f', '-0x1.9906cap-13f'],
            'Cos':              ['0x1.1p-1f', '-0x1.ffffap-2f', '0x1.553ce6p-5f', '-0x1.6424ap-10f'],
            'Tan':              ['0x1.fffa5ap-1f', '0x1.579a9p-2f', '0x1.db9ee4p-4f', '0x1.80ebbap-4f'],
            'Asin':             ['0x1.0001e4p+0f', '0x1.5120dp-3f', '0x1.87ef0cp-4f'],
            'Atan':             ['0x1.ffdf9ap-1f', '-0x1.4d0938p-2f', '0x1.3b964p-3f', '-0x1.633322p-5f'],
            'Sinh':             ['0x1.0p+0f', '0x1.55547cp-3f', '0x1.123d82p-7f'],
            'Cosh':             ['0x1.000002p+0f', '0x1.fffd0ap-2f', '0x1.576362p-5f'],
            'Asinh':            ['0x1.fff2ccp-1f', '-0x1.4d786ap-3f', '0x1.a622bcp-5f'],
            'Acosh':            ['0x1.6a084ep+0f', '-0x1.df194cp-4f', '0x1.611af2p-6f'],
            'Atanh':            ['0x1.00064cp+0f', '0x1.4e542ep-2f', '0x1.12cebcp-2f'],
            'Erf':              ['0x1.20db82p+0f', '-0x1.802cbcp-2f', '0x1.ba0bcep-4f', '-0x1.2dded4p-6f'],
            'ErfcAsymptotic':   ['0x1.0p+0f', '-0x1.0p-1f', '0x1.8p-1f'],
            'SqrtSeed':         ['0x1.669044p-3f', '0x1.433fa4p+0f', '-0x1.ac8a38p-1f', '0x1.df42c2p-2f'],
            'RsqrtSeed':        ['0x1.c6b9cep+0f', '-0x1.67cdcep+0f', '0x1.f3625ap-1f', '-0x1.7d02f2p-2f'],
            'CbrtSeed':         ['0x1.395abp-1f', '0x1.ba4fd4p-1f', '-0x1.23bd2cp-1f', '0x1.18e382p-2f'],
            'Lanczos':          ['0x1.1p-1f', '0x1.52429cp+9f', '-0x1.3ac8e8p+10f', '0x1.81a966p+9f', '-0x1.613ae6p+7f', '0x1.903c28p+3f', '-0x1.1bcb2ap-3f', '0x1.4b045ap-17f'],
        },
    },
    'f64': {
        'balanced': {
            'Exp':              ['0x1.0p+0', '0x1.000000a1fd6a7p+0', '0x1.00000028794dfp-1', '0x1.5554043e289bcp-3', '0x1.5554ace120b86p-5', '0x1.126fa6fd6e815p-7', '0x1.6d7531eae5468p-10'],
            'Expm1':            ['0x1.0p+0', '0x1.00000028794dfp-1', '0x1.5555556750672p-3', '0x1.5554ace120b86p-5', '0x1.1110c63a4eedp-7', '0x1.6d7531eae5468p-10', '0x1.a15169e096556p-13'],
            'Exp2':             ['0x1.0p+0', '0x1.62e4300881c7cp-1', '0x1.ebfbe000ca124p-3', '0x1.c6b025ee4a366p-5', '0x1.3b2a931e876e3p-7', '0x1.5e5f5e0421cbp-10', '0x1.439e545c3ea41p-13'],
            'Exp10':            ['0x1.0p+0', '0x1.26bb89ad792a1p+1', '0x1.53526bffeca71p+1', '0x1.0453c65b5f11ep+1', '0x1.2bc6f9fb54aa7p+0', '0x1.1b92cf22ad3dcp-1', '0x1.b070655a64ecep-3'],
            'Log1p':            ['0x1.8d7991535d116p-18', '0x1.ffff4d94c6107p-1', '-0x1.0062bc78e07f5p-1', '0x1.560698110ba1bp-2', '-0x1.e23434a7ebd19p-3', '0x1.7edf5b3ff3342p-3', '-0x1.01ea76bbc4ecap-2', '0x1.c0cfad98c53b4p-3'],
            'Log2':             ['0x1.1eb7af09a833ap-17', '0x1.7153f59f021bbp+0', '-0x1.71e2e8906e383p-1', '0x1.ed70597005a53p-2', '-0x1.5bd62bf398f84p-2', '0x1.142f2263e4ce6p-2', '-0x1.74180d4905dbp-2', '0x1.43bfb66774ef3p-2'],
            'Log10':            ['0x1.593e06a240da5p-19', '0x1.bcb716596ef52p-2', '-0x1.bd63370d3230bp-3', '0x1.29146dae7a7fap-3', '-0x1.a2d646531efa7p-4', '0x1.4c8f0e906d925p-4', '-0x1.c00bb72eb21a3p-4', '0x1.85d51e35149d8p-4'],
            'Sin':              ['0x1.1p-1', '-0x1.5555555555555p-3', '0x1.1111111111111p-7', '-0x1.a01a01a01a014p-13', '0x1.71de3a556bb7ep-19', '-0x1.ae64567d9d7a9p-26', '0x1.61245edeb306ap-33', '-0x1.ae7b75b7f3ea8p-41', '0x1.91e5d3ff5106bp-49'],
            'Cos':              ['0x1.1p-1', '-0x1.1p-2', '0x1.5555555555555p-5', '-0x1.6c16c16c16c09p-10', '0x1.a01a01a01844fp-16', '-0x1.27e4fb7581302p-22', '0x1.1eed8c32f1021p-29', '-0x1.9392cccc6be36p-37', '0x1.aa9bc439ae3a9p-45'],
            'Tan':              ['0x1.000000005a5b8p+0', '0x1.555553e31f706p-2', '0x1.11118d1f228a7p-3', '0x1.b9fbd9de55dfcp-5', '0x1.68548d37e29e3p-6', '0x1.10332abe2f058p-7', '0x1.49f42392f1bf8p-8', '-0x1.4c5d8d7f2d263p-11', '0x1.0d67c14888fd1p-9'],
            'Asin':             ['0x1.ffffffffe1fddp-1', '0x1.55555644ffd27p-3', '0x1.3332974333413p-4', '0x1.6ddd3297735ebp-5', '0x1.ed3310aa58a51p-6', '0x1.93e1363c82c76p-6', '0x1.e5084c0c9e239p-8', '0x1.1ca8ba846c6cbp-5'],
            'Atan':             ['0x1.ffffffe79bf93p-1', '-0x1.55552f0db432p-2', '0x1.998f82493d066p-3', '-0x1.240b8d994abf9p-3', '0x1.bfc9e8c2b0243p-4', '-0x1.5601364b1cac3p-4', '0x1.d3ad79f53e142p-5', '-0x1.f148f2a829af1p-6', '0x1.57a8c747ed314p-7', '-0x1.bdf39729124b6p-10'],
            'Sinh':             ['0x1.0p+0', '0x1.5555555555555p-3', '0x1.1111111111113p-7', '0x1.a01a01a0185b8p-13', '0x1.71de3a5f3a85fp-19', '0x1.ae641aa4b3bc4p-26', '0x1.61d987751d063p-33'],
            'Cosh':             ['0x1.0p+0', '0x1.1p-2', '0x1.555555555555dp-5', '0x1.6c16c16c13a9dp-10', '0x1.a01a01b27d442p-16', '0x1.27e4c358312d7p-22', '0x1.1f97684759d2cp-29'],
            'Asinh':            ['0x1.ffffffffd31a3p-1', '-0x1.5555546833a3bp-3', '0x1.3332cb1306bap-4', '-0x1.6da50cdef9bd8p-5', '0x1.f03a9698a9a1ap-6', '-0x1.64a1d4e884e18p-6', '0x1.eb17f1de08ap-7', '-0x1.0afc2605b475ap-7', '0x1.38db71d294bacp-9'],
            'Acosh':            ['0x1.6a09e667efe68p+0', '-0x1.e2b7dda271da1p-4', '0x1.b2721effa4bf4p-6', '-0x1.0294304858003p-7', '0x1.5f548154f92bfp-9', '-0x1.fabbd22ca2814p-11', '0x1.5677d4f887a75p-12', '-0x1.2dd8e606c72cbp-14'],
            'Atanh':            ['0x1.ffffffff5e368p-1', '0x1.555557db56886p-2', '0x1.9997f57975c75p-3', '0x1.24c5e33e88f82p-3', '0x1.c0f55975c35e6p-4', '0x1.a66a48ce8cef7p-4', '0x1.859eb5c8eb941p-6', '0x1.69d411adf045dp-3'],
            'Erf':              ['0x1.20dd750429b19p+0', '-0x1.812746b0271ecp-2', '0x1.ce2f2197bc2efp-4', '-0x1.b82ce15c3b7f8p-6', '0x1.565ba0406c001p-8', '-0x1.c0286cad68e02p-11', '0x1.f941bed7e7847p-14', '-0x1.f0705a1dde9aap-17', '0x1.9ae0986f9de4fp-20', '-0x1.c1a515de8637ap-24'],
            'ErfcAsymptotic':   ['0x1.0p+0', '-0x1.0p-1', '0x1.8p-1', '-0x1.ep+0'],
            'SqrtSeed':         ['0x1.669044b27e55fp-3', '0x1.433fa4f9223f1p+0', '-0x1.ac8a37f93cd48p-1', '0x1.df42c1629a518p-2', '-0x1.42c35908092b3p-4'],
            'RsqrtSeed':        ['0x1.c6b9cdfab5492p+0', '-0x1.67cdcdd30e8ddp+0', '0x1.f3625a4e67596p-1', '-0x1.7d02f10610a8p-2', '0x1.cd28767f57645p-5'],
            'CbrtSeed':         ['0x1.395aafeedb4p-1', '0x1.ba4fd4ac561dfp-1', '-0x1.23bd2bd853685p-1', '0x1.18e3815d16328p-2', '-0x1.e2ee2147bdd1fp-5'],
            'Lanczos':          ['0x1.ffffffffff95p-1', '0x1.52429b6c30b05p+9', '-0x1.3ac8e8ed4171bp+10', '0x1.81a9661d3b4d8p+9', '-0x1.613ae51a32f5dp+7', '0x1.903c27f8b9c81p+3', '-0x1.1bcb2992b2855p-3', '0x1.4b045a9e6c012p-17', '0x1.433a1cc831b75p-23'],
        },
        'approximate': {
            'Exp':              ['0x1.0p+0', '0x1.fffb0ed184294p-1', '0x1.fffe5aa63cb93p-2', '0x1.57e730d6db7bbp-3', '0x1.570b976d7eabfp-5'],
            'Expm1':            ['0x1.0p+0', '0x1.fffe5aa63cb93p-2', '0x1.5554dd04e58bep-3', '0x1.570b976d7eabfp-5', '0x1.120b616c43f75p-7'],
            'Exp2':             ['0x1.0p+0', '0x1.62e365e1121f6p-1', '0x1.ebfb82a3aea0bp-3', '0x1.c854f68eba124p-5', '0x1.3beceb44761e5p-7'],
            'Exp10':            ['0x1.0p+0', '0x1.2669e62ee05c6p+1', '0x1.533347ca7eb8bp+1', '0x1.0ef698e2ecd8dp+1', '0x1.33e3775c26ab2p+0'],
            'Log1p':            ['-0x1.0p-138', '0x1.fd72336722531p-1', '-0x1.fbb0efea42193p-2', '0x1.a354556c0709ep-2', '-0x1.41a74ab561c2cp-2'],
            'Log2':             ['-0x1.6dp-138', '0x1.6f7cd883e3903p+0', '-0x1.6e38c5437e568p-1', '0x1.2e7b744e8dd6ep-1', '-0x1.d00c423f00dfcp-2'],
            'Log10':            ['-0x1.34p-140', '0x1.ba7fcf54cef43p-2', '-0x1.b8f9959f4d36ap-3', '0x1.6c3993f2609ap-3', '-0x1.1762797a9d07cp-3'],
            'Sin':              ['0x1.fffffffffffd9p-1', '-0x1.5555555550efep-3', '0x1.1111110bde5e3p-7', '-0x1.a019f8a207d3fp-13', '0x1.71d7317b8ee33p-19', '-0x1.a9507f3711e2dp-26'],
            'Cos':              ['0x1.ffffffffffe0bp-1', '-0x1.ffffffffe3763p-2', '0x1.5555554476425p-5', '-0x1.6c16b2d3e61f5p-10', '0x1.a00e9685da9c4p-16', '-0x1.23c5c15441ap-22'],
            'Tan':              ['0x1.fffff88c9c162p-1', '0x1.555c1aa017c8fp-2', '0x1.10158d12320c6p-3', '0x1.d44a862e47de4p-5', '0x1.9bb3cce75e6a5p-7', '0x1.4db42acf4e25ep-6'],
            'Asin':             ['0x1.00000137fe155p+0', '0x1.554dbf185abeep-3', '0x1.350eaed06a2d5p-4', '0x1.45feff0af4e96p-5', '0x1.9c87b69ffd0f5p-5'],
            'Atan':             ['0x1.ffff52add7cbdp-1', '-0x1.54f2b56f390a5p-2', '0x1.9011df7c714e8p-3', '-0x1.ed5b7ee5714bep-4', '0x1.cf95ccf6dc4c9p-5', '-0x1.ae41a0cc72a89p-7'],
            'Sinh':             ['0x1.0000000000006p+0', '0x1.5555555550d88p-3', '0x1.11111123bf154p-7', '0x1.a01994c849582p-13', '0x1.72e107c874de9p-19'],
            'Cosh':             ['0x1.000000000003dp+0', '0x1.ffffffffe74f1p-2', '0x1.55555588b8403p-5', '0x1.6c162bb7d965cp-10', '0x1.a17df0d914d6cp-16'],
            'Asinh':            ['0x1.fffffeb87136cp-1', '-0x1.55525256d6e19p-3', '0x1.329cfb7132e28p-4', '-0x1.62b33c784e945p-5', '0x1.8f72d64b4487p-6', '-0x1.165b11aa660f5p-7'],
            'Acosh':            ['0x1.6a09e5c26a954p+0', '-0x1.e2b3cf54b0e0dp-4', '0x1.b16970f172561p-6', '-0x1.ecfc95a0a290dp-8', '0x1.cc0297ccbef79p-10'],
            'Atanh':            ['0x1.000005356fbd4p+0', '0x1.5545203965275p-2', '0x1.9d8f88475479ap-3', '0x1.f4c8aeef23538p-4', '0x1.8fc48a7c4bd6dp-3'],
            'Erf':              ['0x1.20dd74546d3e7p+0', '-0x1.812680a1a33f4p-2', '0x1.ce0ab171c3ee9p-4', '-0x1.b5b30f0f42c6cp-6', '0x1.428d76f2cc227p-8', '-0x1.281ef9ed3249cp-11'],
            'ErfcAsymptotic':   ['0x1.0p+0', '-0x1.0p-1', '0x1.8p-1'],
            'SqrtSeed':         ['0x1.669044b27e55fp-3', '0x1.433fa4f9223f1p+0', '-0x1.ac8a37f93cd48p-1', '0x1.df42c1629a518p-2'],
            'RsqrtSeed':        ['0x1.c6b9cdfab5492p+0', '-0x1.67cdcdd30e8ddp+0', '0x1.f3625a4e67596p-1', '-0x1.7d02f10610a8p-2'],
            'CbrtSeed':         ['0x1.395aafeedb4p-1', '0x1.ba4fd4ac561dfp-1', '-0x1.23bd2bd853685p-1', '0x1.18e3815d16328p-2'],
            'Lanczos':          ['0x1.ffffffffff95p-1', '0x1.52429b6c30b05p+9', '-0x1.3ac8e8ed4171bp+10', '0x1.81a9661d3b4d8p+9', '-0x1.613ae51a32f5dp+7', '0x1.903c27f8b9c81p+3', '-0x1.1bcb2992b2855p-3', '0x1.4b045a9e6c012p-17'],
        },
    },
}

STRUCT_ORDER = [
    'Exp', 'Expm1', 'Exp2', 'Exp10',
    'Log1p', 'Log2', 'Log10',
    'Sin', 'Cos', 'Tan',
    'Asin', 'Atan',
    'Sinh', 'Cosh',
    'Asinh', 'Acosh', 'Atanh',
    'Erf', 'ErfcAsymptotic',
    'SqrtSeed', 'RsqrtSeed', 'CbrtSeed',
    'Lanczos',
]

DELETED_SPECIALS_TMPL = [
    '    {n}() = delete;',
    '    ~{n}() = delete;',
    '    {n}(const {n}&) = delete;',
    '    {n}({n}&&) = delete;',
    '    {n}& operator=(const {n}&) = delete;',
    '    {n}& operator=({n}&&) = delete;',
]

def deleted_specials(name):
    return [t.format(n=name) for t in DELETED_SPECIALS_TMPL]

def emit_struct_f32(name, coeffs):
    lines = [f'struct LEIBNIZ_RUNTIME_API {name} final {{']
    lines.extend(deleted_specials(name))
    lines.append('')
    for i, c in enumerate(coeffs):
        lines.append(f'    static inline const __m256& c{i}() {{ static const __m256 v = _mm256_set1_ps({c}); return v; }}')
    lines.append('};')
    return lines

def emit_struct_f64(name, coeffs):
    lines = [f'struct LEIBNIZ_RUNTIME_API {name} final {{']
    lines.extend(deleted_specials(name))
    lines.append('')
    for i, c in enumerate(coeffs):
        lines.append(f'    static inline const __m256d& c{i}() {{ static const __m256d v = _mm256_set1_pd({c}); return v; }}')
    lines.append('};')
    return lines

def generate(output_path):
    out = []
    out.append('// AUTO-GENERATED by striped_tables_gen.py - DO NOT EDIT BY HAND')
    out.append('// Leibniz HPC Math Library - AVX2 Pre-Broadcast Stripe Tables')
    out.append('//')
    out.append('// Purpose:')
    out.append('//   Provides lazy-static __m256 / __m256d broadcast registers for every')
    out.append('//   LookUpTables.h coefficient, eliminating scalar load + broadcast from')
    out.append('//   Horner evaluation hot paths.')
    out.append('//')
    out.append('// Tiers included:')
    out.append('//   Balanced    - Chebyshev minimax, ~N/2 terms')
    out.append('//   Approximate - Chebyshev minimax, ~N/3 terms')
    out.append('//   Exact tier is intentionally excluded (register pressure)')
    out.append('//')
    out.append('// Types:')
    out.append('//   FP32  - __m256  (8 x float,  AVX2)')
    out.append('//   FP64  - __m256d (4 x double, AVX2)')
    out.append('//')
    out.append('// Usage pattern (Horner kernel):')
    out.append('//   Leibniz::Numerics::Vectorized::StripedTables::FP32::Balanced::Exp tbl;')
    out.append('//   __m256 acc = tbl.c3();')
    out.append('//   acc = _mm256_fmadd_ps(acc, x, tbl.c2());')
    out.append('//   acc = _mm256_fmadd_ps(acc, x, tbl.c1());')
    out.append('//   acc = _mm256_fmadd_ps(acc, x, tbl.c0());')
    out.append('//')
    out.append('// Evaluation conventions (inherited from LookUpTables.h):')
    out.append('//   Straight  (Exp, Log*, Expm1, Exp2, Exp10, Acosh):  Horner(x, c)')
    out.append('//   Odd       (Sin, Tan, Asin, Atan, Sinh, Asinh, Atanh, Erf): x * Horner(x^2, c)')
    out.append('//   Even      (Cos, Cosh):                              Horner(x^2, c)')
    out.append('//   Seeds     (SqrtSeed, RsqrtSeed, CbrtSeed):          seed + Newton-Raphson')
    out.append('//   Lanczos   (lgamma/tgamma):                          rational sum g=7, 9-term')
    out.append('//   ErfcAsymptotic: (exp(-x^2) / (x * sqrt(pi))) * Horner(1/x^2, c)')
    out.append('//')
    out.append('// Functions without tables (share stripe registers from above):')
    out.append('//   ACOS      acos(x)  = pi/2 - asin(x)          - reuses Asin stripes')
    out.append('//   ATAN2     atan2(y,x) = atan + quadrant fix     - reuses Atan stripes')
    out.append('//   TANH      tanh(x)  = sinh(x)/cosh(x)          - reuses Sinh, Cosh stripes')
    out.append('//   POW       pow(x,y) = exp(y*log(x))            - reuses Exp, Log1p stripes')
    out.append('//   HYPOT     hypot(a,b) = sqrt(a^2+b^2)          - reuses SqrtSeed stripes')
    out.append('//   SIGMOID   sigmoid(x) = 1/(1+exp(-x))          - reuses Exp stripes')
    out.append('//   SOFTPLUS  softplus(x) = log(1+exp(x))         - reuses Exp, Log1p stripes')
    out.append('//   LGAMMA    lgamma/tgamma share Lanczos stripes')
    out.append('//')
    out.append('// Namespace structure:')
    out.append('//   Leibniz::Numerics::Vectorized::StripedTables::FP32::{Balanced,Approximate}::{Struct}')
    out.append('//   Leibniz::Numerics::Vectorized::StripedTables::FP64::{Balanced,Approximate}::{Struct}')
    out.append('//')
    out.append('// Accessor pattern (lazy static, initialized once on first call):')
    out.append('//   inline const __m256&  cN() { static const __m256  v = _mm256_set1_ps(...); return v; }')
    out.append('//   inline const __m256d& cN() { static const __m256d v = _mm256_set1_pd(...); return v; }')
    out.append('#pragma once')
    out.append('#include <immintrin.h>')
    out.append('')
    out.append('namespace Leibniz::Numerics::Vectorized::StripedTables {')

    for type_key, type_ns, emit_fn in [('f32', 'FP32', emit_struct_f32), ('f64', 'FP64', emit_struct_f64)]:
        out.append('')
        out.append(f'namespace {type_ns} {{')
        for tier_key, tier_ns in [('balanced', 'Balanced'), ('approximate', 'Approximate')]:
            out.append('')
            out.append(f'namespace {tier_ns} {{')
            out.append('')
            tier_data = TABLES[type_key][tier_key]
            for struct_name in STRUCT_ORDER:
                out.extend(emit_fn(struct_name, tier_data[struct_name]))
                out.append('')
            out.append(f'}} // namespace {tier_ns}')
        out.append('')
        out.append(f'}} // namespace {type_ns}')

    out.append('')
    out.append('} // namespace Leibniz::Numerics::Vectorized::StripedTables')

    with open(output_path, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print(f'Written: {output_path}')

if __name__ == '__main__':
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'StripedTables.h'
    generate(output_path)
