import argparse
import json
import os
from tqdm import tqdm
import cv2
import torch
import torch.utils.data
from torchvision import transforms, datasets
import torchvision

from skimage import io, transform
import numpy as np


class ETHECLabelMap:
    """
    Implements map from labels to hot vectors for ETHEC database.
    """
    def __init__(self):
        self.family = {
            "Hesperiidae": 0,
            "Papilionidae": 1,
            "Pieridae": 2,
            "Nymphalidae": 3,
            "Lycaenidae": 4,
            "Riodinidae": 5
        }
        self.subfamily = {
            "Heteropterinae": 0,
            "Hesperiinae": 1,
            "Pyrginae": 2,
            "Parnassiinae": 3,
            "Papilioninae": 4,
            "Dismorphiinae": 5,
            "Coliadinae": 6,
            "Pierinae": 7,
            "Satyrinae": 8,
            "Lycaeninae": 9,
            "Nymphalinae": 10,
            "Heliconiinae": 11,
            "Nemeobiinae": 12,
            "Theclinae": 13,
            "Aphnaeinae": 14,
            "Polyommatinae": 15,
            "Libytheinae": 16,
            "Danainae": 17,
            "Charaxinae": 18,
            "Apaturinae": 19,
            "Limenitidinae": 20
        }
        self.genus = {
            "Carterocephalus": 0,
            "Heteropterus": 1,
            "Thymelicus": 2,
            "Hesperia": 3,
            "Ochlodes": 4,
            "Gegenes": 5,
            "Erynnis": 6,
            "Carcharodus": 7,
            "Spialia": 8,
            "Muschampia": 9,
            "Pyrgus": 10,
            "Parnassius": 11,
            "Archon": 12,
            "Sericinus": 13,
            "Zerynthia": 14,
            "Allancastria": 15,
            "Bhutanitis": 16,
            "Luehdorfia": 17,
            "Papilio": 18,
            "Iphiclides": 19,
            "Leptidea": 20,
            "Colias": 21,
            "Aporia": 22,
            "Catopsilia": 23,
            "Gonepteryx": 24,
            "Sinopieris": 25,
            "Mesapia": 26,
            "Baltia": 27,
            "Pieris": 28,
            "Erebia": 29,
            "Berberia": 30,
            "Paralasa": 31,
            "Proterebia": 32,
            "Boeberia": 33,
            "Loxerebia": 34,
            "Proteerbia": 35,
            "Lycaena": 36,
            "Melitaea": 37,
            "Argynnis": 38,
            "Heliophorus": 39,
            "Cethosia": 40,
            "Childrena": 41,
            "Argyronome": 42,
            "Pontia": 43,
            "Anthocharis": 44,
            "Zegris": 45,
            "Euchloe": 46,
            "Colotis": 47,
            "Hamearis": 48,
            "Polycaena": 49,
            "Favonius": 50,
            "Cigaritis": 51,
            "Tomares": 52,
            "Chrysozephyrus": 53,
            "Ussuriana": 54,
            "Coreana": 55,
            "Japonica": 56,
            "Thecla": 57,
            "Celastrina": 58,
            "Artopoetes": 59,
            "Laeosopis": 60,
            "Callophrys": 61,
            "Zizeeria": 62,
            "Pseudozizeeria": 63,
            "Tarucus": 64,
            "Cyclyrius": 65,
            "Leptotes": 66,
            "Satyrium": 67,
            "Lampides": 68,
            "Azanus": 69,
            "Neolycaena": 70,
            "Cupido": 71,
            "Maculinea": 72,
            "Glaucopsyche": 73,
            "Pseudophilotes": 74,
            "Scolitantides": 75,
            "Iolana": 76,
            "Plebejus": 77,
            "Caerulea": 78,
            "Afarsia": 79,
            "Agriades": 80,
            "Alpherakya": 81,
            "Plebejidea": 82,
            "Kretania": 83,
            "Maurus": 84,
            "Aricia": 85,
            "Pamiria": 86,
            "Polyommatus": 87,
            "Eumedonia": 88,
            "Cyaniris": 89,
            "Lysandra": 90,
            "Glabroculus": 91,
            "Neolysandra": 92,
            "Libythea": 93,
            "Danaus": 94,
            "Charaxes": 95,
            "Mimathyma": 96,
            "Apatura": 97,
            "Limenitis": 98,
            "Euapatura": 99,
            "Hestina": 100,
            "Timelaea": 101,
            "Thaleropis": 102,
            "Parasarpa": 103,
            "Lelecella": 104,
            "Neptis": 105,
            "Nymphalis": 106,
            "Athyma": 107,
            "Inachis": 108,
            "Araschnia": 109,
            "Junonia": 110,
            "Vanessa": 111,
            "Speyeria": 112,
            "Fabriciana": 113,
            "Issoria": 114,
            "Brenthis": 115,
            "Boloria": 116,
            "Kuekenthaliella": 117,
            "Clossiana": 118,
            "Proclossiana": 119,
            "Meiltaea": 120,
            "Euphydryas": 121,
            "Melanargia": 122,
            "Davidina": 123,
            "Hipparchia": 124,
            "Chazara": 125,
            "Pseudochazara": 126,
            "Karanasa": 127,
            "Paroeneis": 128,
            "Oeneis": 129,
            "Satyrus": 130,
            "Minois": 131,
            "Arethusana": 132,
            "Brintesia": 133,
            "Maniola": 134,
            "Aphantopus": 135,
            "Hyponephele": 136,
            "Pyronia": 137,
            "Coenonympha": 138,
            "Pararge": 139,
            "Ypthima": 140,
            "Lasiommata": 141,
            "Lopinga": 142,
            "Kirinia": 143,
            "Neope": 144,
            "Lethe": 145,
            "Mycalesis": 146,
            "Arisbe": 147,
            "Atrophaneura": 148,
            "Agehana": 149,
            "Teinopalpus": 150,
            "Graphium": 151,
            "Meandrusa": 152
        }
        self.specific_epithet = {
            "palaemon": 0,
            "silvicola": 1,
            "morpheus": 2,
            "sylvestris": 3,
            "lineola": 4,
            "hamza": 5,
            "acteon": 6,
            "comma": 7,
            "venata": 8,
            "nostrodamus": 9,
            "tages": 10,
            "alceae": 11,
            "lavatherae": 12,
            "baeticus": 13,
            "floccifera": 14,
            "sertorius": 15,
            "orbifer": 16,
            "cribrellum": 17,
            "proto": 18,
            "tessellum": 19,
            "accretus": 20,
            "alveus": 21,
            "armoricanus": 22,
            "andromedae": 23,
            "cacaliae": 24,
            "carlinae": 25,
            "carthami": 26,
            "malvae": 27,
            "cinarae": 28,
            "cirsii": 29,
            "centaureae": 30,
            "bellieri": 31,
            "malvoides": 32,
            "onopordi": 33,
            "serratulae": 34,
            "sidae": 35,
            "warrenensis": 36,
            "sacerdos": 37,
            "apollinus": 38,
            "apollinaris": 39,
            "apollo": 40,
            "geminus": 41,
            "mnemosyne": 42,
            "glacialis": 43,
            "montela": 44,
            "rumina": 45,
            "polyxena": 46,
            "cerisyi": 47,
            "deyrollei": 48,
            "caucasica": 49,
            "cretica": 50,
            "thaidina": 51,
            "lidderdalii": 52,
            "mansfieldi": 53,
            "japonica": 54,
            "puziloi": 55,
            "chinensis": 56,
            "machaon": 57,
            "stubbendorfii": 58,
            "apollonius": 59,
            "alexanor": 60,
            "hospiton": 61,
            "xuthus": 62,
            "podalirius": 63,
            "feisthamelii": 64,
            "sinapis": 65,
            "palaeno": 66,
            "podalirinus": 67,
            "pelidne": 68,
            "juvernica": 69,
            "morsei": 70,
            "amurensis": 71,
            "duponcheli": 72,
            "marcopolo": 73,
            "ladakensis": 74,
            "grumi": 75,
            "nebulosa": 76,
            "nastes": 77,
            "tamerlana": 78,
            "cocandica": 79,
            "sieversi": 80,
            "sifanica": 81,
            "alpherakii": 82,
            "christophi": 83,
            "shahfuladi": 84,
            "tyche": 85,
            "phicomone": 86,
            "montium": 87,
            "alfacariensis": 88,
            "hyale": 89,
            "erate": 90,
            "erschoffi": 91,
            "romanovi": 92,
            "regia": 93,
            "stoliczkana": 94,
            "hecla": 95,
            "eogene": 96,
            "thisoa": 97,
            "staudingeri": 98,
            "lada": 99,
            "baeckeri": 100,
            "adelaidae": 101,
            "fieldii": 102,
            "heos": 103,
            "diva": 104,
            "chrysotheme": 105,
            "balcanica": 106,
            "myrmidone": 107,
            "croceus": 108,
            "felderi": 109,
            "viluiensis": 110,
            "crataegi": 111,
            "aurorina": 112,
            "chlorocoma": 113,
            "libanotica": 114,
            "wiskotti": 115,
            "florella": 116,
            "rhamni": 117,
            "maxima": 118,
            "sagartia": 119,
            "cleopatra": 120,
            "cleobule": 121,
            "amintha": 122,
            "mahaguru": 123,
            "davidis": 124,
            "procris": 125,
            "hippia": 126,
            "peloria": 127,
            "potanini": 128,
            "nabellica": 129,
            "butleri": 130,
            "shawii": 131,
            "brassicae": 132,
            "cheiranthi": 133,
            "rapae": 134,
            "gorge": 135,
            "aethiopellus": 136,
            "mnestra": 137,
            "epistygne": 138,
            "turanica": 139,
            "ottomana": 140,
            "tyndarus": 141,
            "oeme": 142,
            "lefebvrei": 143,
            "melas": 144,
            "zapateri": 145,
            "neoridas": 146,
            "montana": 147,
            "cassioides": 148,
            "nivalis": 149,
            "scipio": 150,
            "pronoe": 151,
            "styx": 152,
            "meolans": 153,
            "palarica": 154,
            "pandrose": 155,
            "hispania": 156,
            "meta": 157,
            "wanga": 158,
            "theano": 159,
            "erinnyn": 160,
            "lambessanus": 161,
            "abdelkader": 162,
            "disa": 163,
            "rossii": 164,
            "cyclopius": 165,
            "hades": 166,
            "afra": 167,
            "parmenio": 168,
            "saxicola": 169,
            "rondoui": 170,
            "mannii": 171,
            "ergane": 172,
            "krueperi": 173,
            "melete": 174,
            "napi": 175,
            "nesis": 176,
            "thersamon": 177,
            "lampon": 178,
            "solskyi": 179,
            "splendens": 180,
            "candens": 181,
            "ochimus": 182,
            "hippothoe": 183,
            "tityrus": 184,
            "asabinus": 185,
            "thetis": 186,
            "athalia": 187,
            "paphia": 188,
            "tamu": 189,
            "brahma": 190,
            "epicles": 191,
            "androcles": 192,
            "biblis": 193,
            "childreni": 194,
            "ruslana": 195,
            "parthenoides": 196,
            "bryoniae": 197,
            "edusa": 198,
            "daplidice": 199,
            "callidice": 200,
            "thibetana": 201,
            "bambusarum": 202,
            "bieti": 203,
            "scolymus": 204,
            "pyrothoe": 205,
            "eupheme": 206,
            "fausti": 207,
            "simplonia": 208,
            "daphalis": 209,
            "chloridice": 210,
            "belemia": 211,
            "ausonia": 212,
            "tagis": 213,
            "crameri": 214,
            "insularis": 215,
            "orientalis": 216,
            "transcaspica": 217,
            "charlonia": 218,
            "penia": 219,
            "tomyris": 220,
            "falloui": 221,
            "pulverata": 222,
            "gruneri": 223,
            "damone": 224,
            "cardamines": 225,
            "belia": 226,
            "euphenoides": 227,
            "fausta": 228,
            "phisadia": 229,
            "protractus": 230,
            "evagore": 231,
            "lucina": 232,
            "phlaeas": 233,
            "helle": 234,
            "pang": 235,
            "caspius": 236,
            "margelanica": 237,
            "li": 238,
            "dispar": 239,
            "alciphron": 240,
            "virgaureae": 241,
            "kasyapa": 242,
            "Tschamut, Tujetsch": 243,
            "quercus": 244,
            "cilissa": 245,
            "siphax": 246,
            "zohra": 247,
            "allardi": 248,
            "ballus": 249,
            "nogelii": 250,
            "mauretanicus": 251,
            "callimachus": 252,
            "smaragdinus": 253,
            "micahaelis": 254,
            "raphaelis": 255,
            "saepestriata": 256,
            "betulae": 257,
            "argiolus": 258,
            "pryeri": 259,
            "roboris": 260,
            "rubi": 261,
            "knysna": 262,
            "maha": 263,
            "theophrastus": 264,
            "webbianus": 265,
            "balkanica": 266,
            "pirithous": 267,
            "spini": 268,
            "boeticus": 269,
            "w-album": 270,
            "ilicis": 271,
            "pruni": 272,
            "acaciae": 273,
            "esculi": 274,
            "jesous": 275,
            "ledereri": 276,
            "rhymnus": 277,
            "karsandra": 278,
            "avis": 279,
            "pirthous": 280,
            "davidi": 281,
            "minimus": 282,
            "rebeli": 283,
            "arion": 284,
            "alcetas": 285,
            "lorquinii": 286,
            "osiris": 287,
            "argiades": 288,
            "decolorata": 289,
            "melanops": 290,
            "alexis": 291,
            "alcon": 292,
            "teleius": 293,
            "abencerragus": 294,
            "bavius": 295,
            "panoptes": 296,
            "vicrama": 297,
            "baton": 298,
            "nausithous": 299,
            "orion": 300,
            "gigantea": 301,
            "iolas": 302,
            "argus": 303,
            "eversmanni": 304,
            "paphos": 305,
            "coeli": 306,
            "astraea": 307,
            "morgiana": 308,
            "argyrognomon": 309,
            "optilete": 310,
            "devanica": 311,
            "loewii": 312,
            "idas": 313,
            "trappi": 314,
            "pylaon": 315,
            "psylorita": 316,
            "martini": 317,
            "allardii": 318,
            "vogelii": 319,
            "samudra": 320,
            "orbitulus": 321,
            "artaxerxes": 322,
            "omphisa": 323,
            "galathea": 324,
            "glandon": 325,
            "agestis": 326,
            "maracandica": 327,
            "damon": 328,
            "montensis": 329,
            "eumedon": 330,
            "nicias": 331,
            "semiargus": 332,
            "dolus": 333,
            "isaurica": 334,
            "anteros": 335,
            "menalcas": 336,
            "antidolus": 337,
            "phyllis": 338,
            "iphidamon": 339,
            "damonides": 340,
            "poseidon": 341,
            "ripartii": 342,
            "admetus": 343,
            "humedasae": 344,
            "dorylas": 345,
            "thersites": 346,
            "escheri": 347,
            "bellargus": 348,
            "coridon": 349,
            "hispana": 350,
            "albicans": 351,
            "caelestissima": 352,
            "punctifera": 353,
            "nivescens": 354,
            "aedon": 355,
            "myrrha": 356,
            "atys": 357,
            "icarus": 358,
            "caeruleus": 359,
            "elvira": 360,
            "cyane": 361,
            "elbursica": 362,
            "firdussii": 363,
            "golgus": 364,
            "ellisoni": 365,
            "coelestina": 366,
            "corona": 367,
            "amandus": 368,
            "venus": 369,
            "daphnis": 370,
            "eros": 371,
            "celina": 372,
            "celtis": 373,
            "plexippus": 374,
            "chrysippus": 375,
            "jasius": 376,
            "nycteis": 377,
            "iris": 378,
            "ilia": 379,
            "reducta": 380,
            "metis": 381,
            "mirza": 382,
            "albescens": 383,
            "populi": 384,
            "camilla": 385,
            "schrenckii": 386,
            "ionia": 387,
            "albomaculata": 388,
            "sydyi": 389,
            "limenitoides": 390,
            "sappho": 391,
            "alwina": 392,
            "rivularis": 393,
            "antiopa": 394,
            "polychloros": 395,
            "xanthomelas": 396,
            "l-album": 397,
            "urticae": 398,
            "punctata": 399,
            "perius": 400,
            "ichnusa": 401,
            "egea": 402,
            "c-album": 403,
            "io": 404,
            "burejana": 405,
            "levana": 406,
            "canace": 407,
            "c-aureum": 408,
            "rizana": 409,
            "hierta": 410,
            "atalanta": 411,
            "vulcania": 412,
            "virginiensis": 413,
            "indica": 414,
            "cardui": 415,
            "pandora": 416,
            "aglaja": 417,
            "niobe": 418,
            "clara": 419,
            "laodice": 420,
            "adippe": 421,
            "jainadeva": 422,
            "auresiana": 423,
            "nerippe": 424,
            "elisa": 425,
            "lathonia": 426,
            "hecate": 427,
            "daphne": 428,
            "ino": 429,
            "pales": 430,
            "eugenia": 431,
            "sipora": 432,
            "aquilonaris": 433,
            "napaea": 434,
            "selene": 435,
            "eunomia": 436,
            "graeca": 437,
            "thore": 438,
            "dia": 439,
            "euphrosyne": 440,
            "titania": 441,
            "freija": 442,
            "iphigenia": 443,
            "chariclea": 444,
            "cinxia": 445,
            "aetherie": 446,
            "arduinna": 447,
            "phoebe": 448,
            "didyma": 449,
            "varia": 450,
            "aurelia": 451,
            "asteria": 452,
            "diamina": 453,
            "punica": 454,
            "britomartis": 455,
            "fergana": 456,
            "acraeina": 457,
            "trivia": 458,
            "persea": 459,
            "ambigua": 460,
            "deione": 461,
            "chitralensis": 462,
            "saxatilis": 463,
            "minerva": 464,
            "scotosia": 465,
            "maturna": 466,
            "ichnea": 467,
            "cynthia": 468,
            "aurinia": 469,
            "sibirica": 470,
            "iduna": 471,
            "titea": 472,
            "parce": 473,
            "lachesis": 474,
            "provincialis": 475,
            "desfontainii": 476,
            "russiae": 477,
            "larissa": 478,
            "ines": 479,
            "pherusa": 480,
            "occitanica": 481,
            "arge": 482,
            "meridionalis": 483,
            "leda": 484,
            "halimede": 485,
            "lugens": 486,
            "hylata": 487,
            "armandi": 488,
            "semele": 489,
            "briseis": 490,
            "parisatis": 491,
            "stulta": 492,
            "fidia": 493,
            "genava": 494,
            "aristaeus": 495,
            "fagi": 496,
            "wyssii": 497,
            "fatua": 498,
            "statilinus": 499,
            "syriaca": 500,
            "neomiris": 501,
            "azorina": 502,
            "prieuri": 503,
            "bischoffii": 504,
            "enervata": 505,
            "persephone": 506,
            "kaufmanni": 507,
            "hippolyte": 508,
            "pelopea": 509,
            "alpina": 510,
            "beroe": 511,
            "schahrudensis": 512,
            "mniszechii": 513,
            "geyeri": 514,
            "telephassa": 515,
            "anthelea": 516,
            "amalthea": 517,
            "cingovskii": 518,
            "orestes": 519,
            "abramovi": 520,
            "modesta": 521,
            "huebneri": 522,
            "palaearcticus": 523,
            "pumilis": 524,
            "magna": 525,
            "tarpeia": 526,
            "norna": 527,
            "actaea": 528,
            "parthicus": 529,
            "ferula": 530,
            "dryas": 531,
            "arethusa": 532,
            "circe": 533,
            "jurtina": 534,
            "hyperantus": 535,
            "pulchra": 536,
            "pulchella": 537,
            "davendra": 538,
            "cadusia": 539,
            "amardaea": 540,
            "lycaon": 541,
            "nurag": 542,
            "lupina": 543,
            "capella": 544,
            "interposita": 545,
            "tithonus": 546,
            "gardetta": 547,
            "tullia": 548,
            "bathseba": 549,
            "cecilia": 550,
            "corinna": 551,
            "sunbecca": 552,
            "pamphilus": 553,
            "janiroides": 554,
            "dorus": 555,
            "elbana": 556,
            "darwiniana": 557,
            "arcania": 558,
            "aegeria": 559,
            "leander": 560,
            "baldus": 561,
            "iphioides": 562,
            "glycerion": 563,
            "hero": 564,
            "oedippus": 565,
            "mongolica": 566,
            "asterope": 567,
            "xiphioides": 568,
            "xiphia": 569,
            "megera": 570,
            "petropolitana": 571,
            "maera": 572,
            "paramegaera": 573,
            "achine": 574,
            "euryale": 575,
            "roxelana": 576,
            "climene": 577,
            "goschkevitschii": 578,
            "diana": 579,
            "francisca": 580,
            "ligea": 581,
            "sicelis": 582,
            "gotama": 583,
            "eriphyle": 584,
            "manto": 585,
            "epiphron": 586,
            "flavofasciata": 587,
            "bubastis": 588,
            "claudina": 589,
            "christi": 590,
            "pharte": 591,
            "aethiops": 592,
            "melampus": 593,
            "sudetica": 594,
            "neriene": 595,
            "triaria": 596,
            "medusa": 597,
            "alberganus": 598,
            "pluto": 599,
            "farinosa": 600,
            "nevadensis": 601,
            "pheretiades": 602,
            "eurypilus": 603,
            "eversmannii": 604,
            "ariadne": 605,
            "stenosemus": 606,
            "hardwickii": 607,
            "charltonius": 608,
            "imperator": 609,
            "acdestis": 610,
            "cardinal": 611,
            "szechenyii": 612,
            "delphius": 613,
            "maximinus": 614,
            "orleans": 615,
            "augustus": 616,
            "loxias": 617,
            "charltontonius": 618,
            "inopinatus": 619,
            "autocrator": 620,
            "cardinalgebi": 621,
            "patricius": 622,
            "stoliczkanus": 623,
            "nordmanni": 624,
            "simo": 625,
            "bremeri": 626,
            "actius": 627,
            "andreji": 628,
            "cephalus": 629,
            "maharaja": 630,
            "tenedius": 631,
            "acco": 632,
            "boedromius": 633,
            "simonius": 634,
            "tianschanicus": 635,
            "phoebus": 636,
            "honrathi": 637,
            "ruckbeili": 638,
            "epaphus": 639,
            "nomion": 640,
            "jacquemonti": 641,
            "mercurius": 642,
            "tibetanus": 643,
            "clodius": 644,
            "smintheus": 645,
            "behrii": 646,
            "mullah": 647,
            "mencius": 648,
            "plutonius": 649,
            "dehaani": 650,
            "polytes": 651,
            "horishana": 652,
            "bootes": 653,
            "elwesi": 654,
            "maackii": 655,
            "impediens": 656,
            "polyeuctes": 657,
            "mandarinus": 658,
            "parus": 659,
            "alcinous": 660,
            "alebion": 661,
            "helenus": 662,
            "imperialis": 663,
            "memnon": 664,
            "eurous": 665,
            "sarpedon": 666,
            "doson": 667,
            "tamerlanus": 668,
            "bianor": 669,
            "paris": 670,
            "hopponis": 671,
            "nevilli": 672,
            "krishna": 673,
            "macilentus": 674,
            "leechi": 675,
            "protenor": 676,
            "cloanthus": 677,
            "thaiwanus": 678,
            "chaon": 679,
            "castor": 680,
            "sciron": 681,
            "arcturus": 682,
            "aureus": 683,
            "lehanus": 684,
            "dieckmanni": 685
        }
        self.genus_specific_epithet = {
            "Carterocephalus_palaemon": 0,
            "Carterocephalus_silvicola": 1,
            "Heteropterus_morpheus": 2,
            "Thymelicus_sylvestris": 3,
            "Thymelicus_lineola": 4,
            "Thymelicus_hamza": 5,
            "Thymelicus_acteon": 6,
            "Hesperia_comma": 7,
            "Ochlodes_venata": 8,
            "Gegenes_nostrodamus": 9,
            "Erynnis_tages": 10,
            "Carcharodus_alceae": 11,
            "Carcharodus_lavatherae": 12,
            "Carcharodus_baeticus": 13,
            "Carcharodus_floccifera": 14,
            "Spialia_sertorius": 15,
            "Spialia_orbifer": 16,
            "Muschampia_cribrellum": 17,
            "Muschampia_proto": 18,
            "Muschampia_tessellum": 19,
            "Pyrgus_accretus": 20,
            "Pyrgus_alveus": 21,
            "Pyrgus_armoricanus": 22,
            "Pyrgus_andromedae": 23,
            "Pyrgus_cacaliae": 24,
            "Pyrgus_carlinae": 25,
            "Pyrgus_carthami": 26,
            "Pyrgus_malvae": 27,
            "Pyrgus_cinarae": 28,
            "Pyrgus_cirsii": 29,
            "Pyrgus_centaureae": 30,
            "Pyrgus_bellieri": 31,
            "Pyrgus_malvoides": 32,
            "Pyrgus_onopordi": 33,
            "Pyrgus_serratulae": 34,
            "Pyrgus_sidae": 35,
            "Pyrgus_warrenensis": 36,
            "Parnassius_sacerdos": 37,
            "Archon_apollinus": 38,
            "Archon_apollinaris": 39,
            "Parnassius_apollo": 40,
            "Parnassius_geminus": 41,
            "Parnassius_mnemosyne": 42,
            "Parnassius_glacialis": 43,
            "Sericinus_montela": 44,
            "Zerynthia_rumina": 45,
            "Zerynthia_polyxena": 46,
            "Allancastria_cerisyi": 47,
            "Allancastria_deyrollei": 48,
            "Allancastria_caucasica": 49,
            "Allancastria_cretica": 50,
            "Bhutanitis_thaidina": 51,
            "Bhutanitis_lidderdalii": 52,
            "Bhutanitis_mansfieldi": 53,
            "Luehdorfia_japonica": 54,
            "Luehdorfia_puziloi": 55,
            "Luehdorfia_chinensis": 56,
            "Papilio_machaon": 57,
            "Parnassius_stubbendorfii": 58,
            "Parnassius_apollonius": 59,
            "Papilio_alexanor": 60,
            "Papilio_hospiton": 61,
            "Papilio_xuthus": 62,
            "Iphiclides_podalirius": 63,
            "Iphiclides_feisthamelii": 64,
            "Leptidea_sinapis": 65,
            "Colias_palaeno": 66,
            "Iphiclides_podalirinus": 67,
            "Colias_pelidne": 68,
            "Leptidea_juvernica": 69,
            "Leptidea_morsei": 70,
            "Leptidea_amurensis": 71,
            "Leptidea_duponcheli": 72,
            "Colias_marcopolo": 73,
            "Colias_ladakensis": 74,
            "Colias_grumi": 75,
            "Colias_nebulosa": 76,
            "Colias_nastes": 77,
            "Colias_tamerlana": 78,
            "Colias_cocandica": 79,
            "Colias_sieversi": 80,
            "Colias_sifanica": 81,
            "Colias_alpherakii": 82,
            "Colias_christophi": 83,
            "Colias_shahfuladi": 84,
            "Colias_tyche": 85,
            "Colias_phicomone": 86,
            "Colias_montium": 87,
            "Colias_alfacariensis": 88,
            "Colias_hyale": 89,
            "Colias_erate": 90,
            "Colias_erschoffi": 91,
            "Colias_romanovi": 92,
            "Colias_regia": 93,
            "Colias_stoliczkana": 94,
            "Colias_hecla": 95,
            "Colias_eogene": 96,
            "Colias_thisoa": 97,
            "Colias_staudingeri": 98,
            "Colias_lada": 99,
            "Colias_baeckeri": 100,
            "Colias_adelaidae": 101,
            "Colias_fieldii": 102,
            "Colias_heos": 103,
            "Colias_caucasica": 104,
            "Colias_diva": 105,
            "Colias_chrysotheme": 106,
            "Colias_balcanica": 107,
            "Colias_myrmidone": 108,
            "Colias_croceus": 109,
            "Colias_felderi": 110,
            "Colias_viluiensis": 111,
            "Aporia_crataegi": 112,
            "Colias_aurorina": 113,
            "Colias_chlorocoma": 114,
            "Colias_libanotica": 115,
            "Colias_wiskotti": 116,
            "Catopsilia_florella": 117,
            "Gonepteryx_rhamni": 118,
            "Gonepteryx_maxima": 119,
            "Colias_sagartia": 120,
            "Gonepteryx_cleopatra": 121,
            "Gonepteryx_cleobule": 122,
            "Gonepteryx_amintha": 123,
            "Gonepteryx_mahaguru": 124,
            "Sinopieris_davidis": 125,
            "Sinopieris_venata": 126,
            "Aporia_procris": 127,
            "Aporia_hippia": 128,
            "Mesapia_peloria": 129,
            "Aporia_potanini": 130,
            "Aporia_nabellica": 131,
            "Baltia_butleri": 132,
            "Baltia_shawii": 133,
            "Pieris_brassicae": 134,
            "Pieris_cheiranthi": 135,
            "Pieris_rapae": 136,
            "Erebia_gorge": 137,
            "Erebia_aethiopellus": 138,
            "Erebia_mnestra": 139,
            "Erebia_epistygne": 140,
            "Erebia_turanica": 141,
            "Erebia_ottomana": 142,
            "Erebia_tyndarus": 143,
            "Erebia_oeme": 144,
            "Erebia_lefebvrei": 145,
            "Erebia_melas": 146,
            "Erebia_zapateri": 147,
            "Erebia_neoridas": 148,
            "Erebia_montana": 149,
            "Erebia_cassioides": 150,
            "Erebia_nivalis": 151,
            "Erebia_scipio": 152,
            "Erebia_pronoe": 153,
            "Erebia_styx": 154,
            "Erebia_meolans": 155,
            "Erebia_palarica": 156,
            "Erebia_pandrose": 157,
            "Erebia_hispania": 158,
            "Erebia_meta": 159,
            "Erebia_wanga": 160,
            "Erebia_theano": 161,
            "Erebia_erinnyn": 162,
            "Berberia_lambessanus": 163,
            "Berberia_abdelkader": 164,
            "Erebia_disa": 165,
            "Erebia_rossii": 166,
            "Erebia_cyclopius": 167,
            "Paralasa_hades": 168,
            "Proterebia_afra": 169,
            "Boeberia_parmenio": 170,
            "Loxerebia_saxicola": 171,
            "Proteerbia_afra": 172,
            "Erebia_rondoui": 173,
            "Pieris_mannii": 174,
            "Pieris_ergane": 175,
            "Pieris_krueperi": 176,
            "Pieris_melete": 177,
            "Pieris_napi": 178,
            "Pieris_nesis": 179,
            "Lycaena_thersamon": 180,
            "Lycaena_lampon": 181,
            "Lycaena_solskyi": 182,
            "Lycaena_splendens": 183,
            "Lycaena_candens": 184,
            "Lycaena_ochimus": 185,
            "Lycaena_hippothoe": 186,
            "Lycaena_tityrus": 187,
            "Lycaena_asabinus": 188,
            "Lycaena_thetis": 189,
            "Melitaea_athalia": 190,
            "Argynnis_paphia": 191,
            "Heliophorus_tamu": 192,
            "Heliophorus_brahma": 193,
            "Heliophorus_epicles": 194,
            "Heliophorus_androcles": 195,
            "Cethosia_biblis": 196,
            "Childrena_childreni": 197,
            "Argyronome_ruslana": 198,
            "Melitaea_parthenoides": 199,
            "Pieris_bryoniae": 200,
            "Pontia_edusa": 201,
            "Pontia_daplidice": 202,
            "Pontia_callidice": 203,
            "Anthocharis_thibetana": 204,
            "Anthocharis_bambusarum": 205,
            "Anthocharis_bieti": 206,
            "Anthocharis_scolymus": 207,
            "Zegris_pyrothoe": 208,
            "Zegris_eupheme": 209,
            "Zegris_fausti": 210,
            "Euchloe_simplonia": 211,
            "Euchloe_daphalis": 212,
            "Pontia_chloridice": 213,
            "Euchloe_belemia": 214,
            "Euchloe_ausonia": 215,
            "Euchloe_tagis": 216,
            "Euchloe_crameri": 217,
            "Euchloe_insularis": 218,
            "Euchloe_orientalis": 219,
            "Euchloe_transcaspica": 220,
            "Euchloe_charlonia": 221,
            "Euchloe_penia": 222,
            "Euchloe_tomyris": 223,
            "Euchloe_falloui": 224,
            "Euchloe_pulverata": 225,
            "Anthocharis_gruneri": 226,
            "Anthocharis_damone": 227,
            "Anthocharis_cardamines": 228,
            "Anthocharis_belia": 229,
            "Anthocharis_euphenoides": 230,
            "Colotis_fausta": 231,
            "Colotis_phisadia": 232,
            "Colotis_protractus": 233,
            "Colotis_evagore": 234,
            "Hamearis_lucina": 235,
            "Polycaena_tamerlana": 236,
            "Lycaena_phlaeas": 237,
            "Lycaena_helle": 238,
            "Lycaena_pang": 239,
            "Lycaena_caspius": 240,
            "Lycaena_margelanica": 241,
            "Lycaena_li": 242,
            "Lycaena_dispar": 243,
            "Lycaena_alciphron": 244,
            "Lycaena_virgaureae": 245,
            "Lycaena_kasyapa": 246,
            "Lycaena_Tschamut, Tujetsch": 247,
            "Favonius_quercus": 248,
            "Cigaritis_cilissa": 249,
            "Cigaritis_siphax": 250,
            "Cigaritis_zohra": 251,
            "Cigaritis_allardi": 252,
            "Tomares_ballus": 253,
            "Tomares_nogelii": 254,
            "Tomares_mauretanicus": 255,
            "Tomares_romanovi": 256,
            "Tomares_callimachus": 257,
            "Chrysozephyrus_smaragdinus": 258,
            "Ussuriana_micahaelis": 259,
            "Coreana_raphaelis": 260,
            "Japonica_saepestriata": 261,
            "Thecla_betulae": 262,
            "Celastrina_argiolus": 263,
            "Artopoetes_pryeri": 264,
            "Laeosopis_roboris": 265,
            "Callophrys_rubi": 266,
            "Zizeeria_knysna": 267,
            "Pseudozizeeria_maha": 268,
            "Tarucus_theophrastus": 269,
            "Cyclyrius_webbianus": 270,
            "Tarucus_balkanica": 271,
            "Leptotes_pirithous": 272,
            "Satyrium_spini": 273,
            "Lampides_boeticus": 274,
            "Satyrium_w-album": 275,
            "Satyrium_ilicis": 276,
            "Satyrium_pruni": 277,
            "Satyrium_acaciae": 278,
            "Satyrium_esculi": 279,
            "Azanus_jesous": 280,
            "Satyrium_ledereri": 281,
            "Neolycaena_rhymnus": 282,
            "Zizeeria_karsandra": 283,
            "Callophrys_avis": 284,
            "Leptotes_pirthous": 285,
            "Neolycaena_davidi": 286,
            "Cupido_minimus": 287,
            "Maculinea_rebeli": 288,
            "Maculinea_arion": 289,
            "Cupido_alcetas": 290,
            "Cupido_lorquinii": 291,
            "Cupido_osiris": 292,
            "Cupido_argiades": 293,
            "Cupido_decolorata": 294,
            "Cupido_staudingeri": 295,
            "Glaucopsyche_melanops": 296,
            "Glaucopsyche_alexis": 297,
            "Maculinea_alcon": 298,
            "Maculinea_teleius": 299,
            "Pseudophilotes_abencerragus": 300,
            "Pseudophilotes_bavius": 301,
            "Pseudophilotes_panoptes": 302,
            "Pseudophilotes_vicrama": 303,
            "Pseudophilotes_baton": 304,
            "Maculinea_nausithous": 305,
            "Scolitantides_orion": 306,
            "Iolana_gigantea": 307,
            "Iolana_iolas": 308,
            "Plebejus_argus": 309,
            "Plebejus_eversmanni": 310,
            "Glaucopsyche_paphos": 311,
            "Caerulea_coeli": 312,
            "Glaucopsyche_astraea": 313,
            "Afarsia_morgiana": 314,
            "Plebejus_argyrognomon": 315,
            "Agriades_optilete": 316,
            "Alpherakya_devanica": 317,
            "Plebejidea_loewii": 318,
            "Plebejus_idas": 319,
            "Kretania_trappi": 320,
            "Kretania_pylaon": 321,
            "Kretania_psylorita": 322,
            "Kretania_martini": 323,
            "Kretania_allardii": 324,
            "Maurus_vogelii": 325,
            "Plebejus_samudra": 326,
            "Agriades_orbitulus": 327,
            "Aricia_artaxerxes": 328,
            "Pamiria_omphisa": 329,
            "Pamiria_galathea": 330,
            "Agriades_glandon": 331,
            "Aricia_agestis": 332,
            "Plebejus_maracandica": 333,
            "Polyommatus_damon": 334,
            "Aricia_montensis": 335,
            "Eumedonia_eumedon": 336,
            "Aricia_nicias": 337,
            "Cyaniris_semiargus": 338,
            "Polyommatus_dolus": 339,
            "Aricia_isaurica": 340,
            "Aricia_anteros": 341,
            "Polyommatus_menalcas": 342,
            "Polyommatus_antidolus": 343,
            "Polyommatus_phyllis": 344,
            "Polyommatus_iphidamon": 345,
            "Polyommatus_damonides": 346,
            "Polyommatus_poseidon": 347,
            "Polyommatus_damone": 348,
            "Polyommatus_ripartii": 349,
            "Polyommatus_admetus": 350,
            "Polyommatus_humedasae": 351,
            "Polyommatus_dorylas": 352,
            "Polyommatus_erschoffi": 353,
            "Polyommatus_thersites": 354,
            "Polyommatus_escheri": 355,
            "Lysandra_bellargus": 356,
            "Lysandra_coridon": 357,
            "Lysandra_hispana": 358,
            "Lysandra_albicans": 359,
            "Lysandra_caelestissima": 360,
            "Lysandra_punctifera": 361,
            "Polyommatus_nivescens": 362,
            "Polyommatus_aedon": 363,
            "Polyommatus_myrrha": 364,
            "Polyommatus_atys": 365,
            "Polyommatus_icarus": 366,
            "Polyommatus_caeruleus": 367,
            "Glabroculus_elvira": 368,
            "Glabroculus_cyane": 369,
            "Polyommatus_elbursica": 370,
            "Polyommatus_firdussii": 371,
            "Polyommatus_stoliczkana": 372,
            "Polyommatus_golgus": 373,
            "Neolysandra_ellisoni": 374,
            "Neolysandra_coelestina": 375,
            "Neolysandra_corona": 376,
            "Polyommatus_amandus": 377,
            "Polyommatus_venus": 378,
            "Polyommatus_daphnis": 379,
            "Polyommatus_eros": 380,
            "Polyommatus_celina": 381,
            "Libythea_celtis": 382,
            "Danaus_plexippus": 383,
            "Danaus_chrysippus": 384,
            "Charaxes_jasius": 385,
            "Mimathyma_nycteis": 386,
            "Apatura_iris": 387,
            "Apatura_ilia": 388,
            "Limenitis_reducta": 389,
            "Apatura_metis": 390,
            "Euapatura_mirza": 391,
            "Hestina_japonica": 392,
            "Timelaea_albescens": 393,
            "Limenitis_populi": 394,
            "Limenitis_camilla": 395,
            "Mimathyma_schrenckii": 396,
            "Thaleropis_ionia": 397,
            "Parasarpa_albomaculata": 398,
            "Limenitis_sydyi": 399,
            "Lelecella_limenitoides": 400,
            "Neptis_sappho": 401,
            "Neptis_alwina": 402,
            "Neptis_rivularis": 403,
            "Nymphalis_antiopa": 404,
            "Nymphalis_polychloros": 405,
            "Nymphalis_xanthomelas": 406,
            "Nymphalis_l-album": 407,
            "Nymphalis_urticae": 408,
            "Athyma_punctata": 409,
            "Athyma_perius": 410,
            "Neptis_pryeri": 411,
            "Nymphalis_ichnusa": 412,
            "Nymphalis_ladakensis": 413,
            "Nymphalis_egea": 414,
            "Nymphalis_c-album": 415,
            "Inachis_io": 416,
            "Araschnia_burejana": 417,
            "Araschnia_levana": 418,
            "Nymphalis_canace": 419,
            "Nymphalis_c-aureum": 420,
            "Nymphalis_rizana": 421,
            "Junonia_hierta": 422,
            "Vanessa_atalanta": 423,
            "Vanessa_vulcania": 424,
            "Vanessa_virginiensis": 425,
            "Vanessa_indica": 426,
            "Vanessa_cardui": 427,
            "Argynnis_pandora": 428,
            "Speyeria_aglaja": 429,
            "Fabriciana_niobe": 430,
            "Speyeria_clara": 431,
            "Argyronome_laodice": 432,
            "Fabriciana_adippe": 433,
            "Fabriciana_jainadeva": 434,
            "Fabriciana_auresiana": 435,
            "Fabriciana_nerippe": 436,
            "Fabriciana_elisa": 437,
            "Issoria_lathonia": 438,
            "Brenthis_hecate": 439,
            "Brenthis_daphne": 440,
            "Brenthis_ino": 441,
            "Boloria_pales": 442,
            "Kuekenthaliella_eugenia": 443,
            "Boloria_sipora": 444,
            "Boloria_aquilonaris": 445,
            "Boloria_napaea": 446,
            "Clossiana_selene": 447,
            "Proclossiana_eunomia": 448,
            "Boloria_graeca": 449,
            "Clossiana_thore": 450,
            "Clossiana_dia": 451,
            "Clossiana_euphrosyne": 452,
            "Clossiana_titania": 453,
            "Clossiana_freija": 454,
            "Clossiana_iphigenia": 455,
            "Clossiana_chariclea": 456,
            "Melitaea_cinxia": 457,
            "Melitaea_aetherie": 458,
            "Melitaea_arduinna": 459,
            "Melitaea_phoebe": 460,
            "Melitaea_didyma": 461,
            "Melitaea_varia": 462,
            "Melitaea_aurelia": 463,
            "Melitaea_asteria": 464,
            "Melitaea_diamina": 465,
            "Meiltaea_didyma": 466,
            "Melitaea_punica": 467,
            "Melitaea_britomartis": 468,
            "Melitaea_fergana": 469,
            "Melitaea_acraeina": 470,
            "Melitaea_trivia": 471,
            "Melitaea_persea": 472,
            "Melitaea_ambigua": 473,
            "Melitaea_deione": 474,
            "Melitaea_chitralensis": 475,
            "Melitaea_saxatilis": 476,
            "Melitaea_turanica": 477,
            "Melitaea_minerva": 478,
            "Melitaea_scotosia": 479,
            "Euphydryas_maturna": 480,
            "Euphydryas_ichnea": 481,
            "Euphydryas_cynthia": 482,
            "Euphydryas_aurinia": 483,
            "Euphydryas_sibirica": 484,
            "Euphydryas_iduna": 485,
            "Melanargia_titea": 486,
            "Melanargia_parce": 487,
            "Melanargia_lachesis": 488,
            "Melanargia_galathea": 489,
            "Euphydryas_provincialis": 490,
            "Euphydryas_desfontainii": 491,
            "Melanargia_russiae": 492,
            "Melanargia_larissa": 493,
            "Melanargia_ines": 494,
            "Melanargia_pherusa": 495,
            "Melanargia_occitanica": 496,
            "Melanargia_arge": 497,
            "Melanargia_meridionalis": 498,
            "Melanargia_leda": 499,
            "Melanargia_halimede": 500,
            "Melanargia_lugens": 501,
            "Melanargia_hylata": 502,
            "Davidina_armandi": 503,
            "Hipparchia_semele": 504,
            "Chazara_briseis": 505,
            "Hipparchia_parisatis": 506,
            "Hipparchia_stulta": 507,
            "Hipparchia_fidia": 508,
            "Hipparchia_genava": 509,
            "Hipparchia_aristaeus": 510,
            "Hipparchia_fagi": 511,
            "Hipparchia_wyssii": 512,
            "Hipparchia_fatua": 513,
            "Hipparchia_statilinus": 514,
            "Hipparchia_syriaca": 515,
            "Hipparchia_neomiris": 516,
            "Hipparchia_azorina": 517,
            "Chazara_prieuri": 518,
            "Chazara_bischoffii": 519,
            "Chazara_enervata": 520,
            "Chazara_persephone": 521,
            "Chazara_kaufmanni": 522,
            "Pseudochazara_hippolyte": 523,
            "Pseudochazara_pelopea": 524,
            "Pseudochazara_alpina": 525,
            "Pseudochazara_beroe": 526,
            "Pseudochazara_schahrudensis": 527,
            "Pseudochazara_mniszechii": 528,
            "Pseudochazara_geyeri": 529,
            "Pseudochazara_telephassa": 530,
            "Pseudochazara_anthelea": 531,
            "Pseudochazara_amalthea": 532,
            "Pseudochazara_graeca": 533,
            "Pseudochazara_cingovskii": 534,
            "Pseudochazara_orestes": 535,
            "Karanasa_abramovi": 536,
            "Karanasa_modesta": 537,
            "Karanasa_huebneri": 538,
            "Paroeneis_palaearcticus": 539,
            "Paroeneis_pumilis": 540,
            "Oeneis_magna": 541,
            "Oeneis_tarpeia": 542,
            "Oeneis_glacialis": 543,
            "Oeneis_norna": 544,
            "Satyrus_actaea": 545,
            "Satyrus_parthicus": 546,
            "Satyrus_ferula": 547,
            "Minois_dryas": 548,
            "Arethusana_arethusa": 549,
            "Brintesia_circe": 550,
            "Maniola_jurtina": 551,
            "Aphantopus_hyperantus": 552,
            "Hyponephele_pulchra": 553,
            "Hyponephele_pulchella": 554,
            "Hyponephele_davendra": 555,
            "Hyponephele_cadusia": 556,
            "Hyponephele_amardaea": 557,
            "Hyponephele_lycaon": 558,
            "Maniola_nurag": 559,
            "Hyponephele_lupina": 560,
            "Hyponephele_capella": 561,
            "Hyponephele_interposita": 562,
            "Pyronia_tithonus": 563,
            "Coenonympha_gardetta": 564,
            "Coenonympha_tullia": 565,
            "Pyronia_bathseba": 566,
            "Pyronia_cecilia": 567,
            "Coenonympha_corinna": 568,
            "Coenonympha_sunbecca": 569,
            "Coenonympha_pamphilus": 570,
            "Pyronia_janiroides": 571,
            "Coenonympha_dorus": 572,
            "Coenonympha_elbana": 573,
            "Coenonympha_darwiniana": 574,
            "Coenonympha_arcania": 575,
            "Pararge_aegeria": 576,
            "Coenonympha_leander": 577,
            "Coenonympha_orientalis": 578,
            "Ypthima_baldus": 579,
            "Coenonympha_iphioides": 580,
            "Coenonympha_glycerion": 581,
            "Coenonympha_hero": 582,
            "Coenonympha_oedippus": 583,
            "Coenonympha_mongolica": 584,
            "Ypthima_asterope": 585,
            "Pararge_xiphioides": 586,
            "Pararge_xiphia": 587,
            "Lasiommata_megera": 588,
            "Lasiommata_petropolitana": 589,
            "Lasiommata_maera": 590,
            "Lasiommata_paramegaera": 591,
            "Lopinga_achine": 592,
            "Erebia_euryale": 593,
            "Kirinia_roxelana": 594,
            "Kirinia_climene": 595,
            "Neope_goschkevitschii": 596,
            "Lethe_diana": 597,
            "Mycalesis_francisca": 598,
            "Erebia_ligea": 599,
            "Lethe_sicelis": 600,
            "Mycalesis_gotama": 601,
            "Kirinia_eversmanni": 602,
            "Erebia_eriphyle": 603,
            "Erebia_manto": 604,
            "Erebia_epiphron": 605,
            "Erebia_flavofasciata": 606,
            "Erebia_bubastis": 607,
            "Erebia_claudina": 608,
            "Erebia_christi": 609,
            "Erebia_pharte": 610,
            "Erebia_aethiops": 611,
            "Erebia_melampus": 612,
            "Erebia_sudetica": 613,
            "Erebia_neriene": 614,
            "Erebia_triaria": 615,
            "Erebia_medusa": 616,
            "Erebia_alberganus": 617,
            "Erebia_pluto": 618,
            "Gonepteryx_farinosa": 619,
            "Melitaea_nevadensis": 620,
            "Agriades_pheretiades": 621,
            "Kretania_eurypilus": 622,
            "Parnassius_eversmannii": 623,
            "Parnassius_ariadne": 624,
            "Parnassius_stenosemus": 625,
            "Parnassius_hardwickii": 626,
            "Parnassius_charltonius": 627,
            "Parnassius_imperator": 628,
            "Parnassius_acdestis": 629,
            "Parnassius_cardinal": 630,
            "Parnassius_szechenyii": 631,
            "Parnassius_delphius": 632,
            "Parnassius_maximinus": 633,
            "Parnassius_staudingeri": 634,
            "Parnassius_orleans": 635,
            "Parnassius_augustus": 636,
            "Parnassius_loxias": 637,
            "Parnassius_charltontonius": 638,
            "Parnassius_inopinatus": 639,
            "Parnassius_autocrator": 640,
            "Parnassius_cardinalgebi": 641,
            "Parnassius_patricius": 642,
            "Parnassius_stoliczkanus": 643,
            "Parnassius_nordmanni": 644,
            "Parnassius_simo": 645,
            "Parnassius_bremeri": 646,
            "Parnassius_actius": 647,
            "Parnassius_andreji": 648,
            "Parnassius_cephalus": 649,
            "Parnassius_maharaja": 650,
            "Parnassius_tenedius": 651,
            "Parnassius_acco": 652,
            "Parnassius_boedromius": 653,
            "Parnassius_simonius": 654,
            "Parnassius_tianschanicus": 655,
            "Parnassius_phoebus": 656,
            "Parnassius_honrathi": 657,
            "Parnassius_ruckbeili": 658,
            "Parnassius_epaphus": 659,
            "Parnassius_nomion": 660,
            "Parnassius_jacquemonti": 661,
            "Parnassius_mercurius": 662,
            "Parnassius_tibetanus": 663,
            "Parnassius_clodius": 664,
            "Parnassius_smintheus": 665,
            "Parnassius_behrii": 666,
            "Arisbe_mullah": 667,
            "Atrophaneura_mencius": 668,
            "Atrophaneura_plutonius": 669,
            "Papilio_dehaani": 670,
            "Papilio_polytes": 671,
            "Atrophaneura_horishana": 672,
            "Papilio_bootes": 673,
            "Agehana_elwesi": 674,
            "Papilio_maackii": 675,
            "Atrophaneura_impediens": 676,
            "Atrophaneura_polyeuctes": 677,
            "Arisbe_mandarinus": 678,
            "Arisbe_parus": 679,
            "Atrophaneura_alcinous": 680,
            "Arisbe_alebion": 681,
            "Papilio_helenus": 682,
            "Teinopalpus_imperialis": 683,
            "Papilio_memnon": 684,
            "Arisbe_eurous": 685,
            "Graphium_sarpedon": 686,
            "Arisbe_doson": 687,
            "Arisbe_tamerlanus": 688,
            "Papilio_bianor": 689,
            "Papilio_paris": 690,
            "Papilio_hopponis": 691,
            "Atrophaneura_nevilli": 692,
            "Papilio_krishna": 693,
            "Papilio_macilentus": 694,
            "Arisbe_leechi": 695,
            "Papilio_protenor": 696,
            "Graphium_cloanthus": 697,
            "Papilio_thaiwanus": 698,
            "Papilio_chaon": 699,
            "Papilio_castor": 700,
            "Meandrusa_sciron": 701,
            "Papilio_arcturus": 702,
            "Teinopalpus_aureus": 703,
            "Agriades_lehanus": 704,
            "Carterocephalus_dieckmanni": 705
        }
        self.levels = [len(self.family), len(self.subfamily), len(self.genus), len(self.specific_epithet)]
        self.n_classes = sum(self.levels)
        self.classes = [key for class_list in [self.family, self.subfamily, self.genus, self.specific_epithet] for key in class_list]
        self.level_names = ['family', 'subfamily', 'genus', 'specific_epithet']

    def get_one_hot(self, family, subfamily, genus, specific_epithet):
        retval = np.zeros(self.n_classes)
        retval[self.family[family]] = 1
        retval[self.subfamily[subfamily] + self.levels[0]] = 1
        retval[self.genus[genus] + self.levels[1]] = 1
        retval[self.specific_epithet[specific_epithet] + self.levels[2]] = 1
        return retval

    def get_label_id(self, level_name, label_name):
        return getattr(self, level_name)[label_name]


class ETHEC:
    """
    ETHEC iterator.
    """
    def __init__(self, path_to_json):
        """
        Constructor.
        :param path_to_json: <str> .json path used for loading database entries.
        """
        self.path_to_json = path_to_json
        with open(path_to_json) as json_file:
            self.data_dict = json.load(json_file)
        self.data_tokens = [token for token in self.data_dict]

    def __getitem__(self, item):
        """
        Fetch an entry based on index.
        :param item: <int> index for the entry in database
        :return: see schema.md
        """
        return self.data_dict[self.data_tokens[item]]

    def __len__(self):
        """
        Returns the number of entries in the database.
        :return: <int> Length of database
        """
        return len(self.data_tokens)

    def get_sample(self, token):
        """
        Fetch an entry based on its token.
        :param token: <str> token (uuid)
        :return: see schema.md
        """
        return self.data_dict[token]


class ETHECDB(torch.utils.data.Dataset):
    """
    Creates a PyTorch dataset.
    """
    def __init__(self, path_to_json, path_to_images, labelmap, transform=None):
        """
        Constructor.
        :param path_to_json: <str> Path to .json from which to read database entries.
        :param path_to_images: <str> Path to parent directory where images are stored.
        :param labelmap: <ETHECLabelMap> Labelmap.
        :param transform: <torchvision.transforms> Set of transforms to be applied to the entries in the database.
        """
        self.path_to_json = path_to_json
        self.path_to_images = path_to_images
        self.labelmap = labelmap
        self.ETHEC = ETHEC(self.path_to_json)
        self.transform = transform

    def __getitem__(self, item):
        """
        Fetch an entry based on index.
        :param item: <int> Index to fetch.
        :return: <dict> Consumable object (see schema.md)
                {'image': <np.array> image, 'labels': <np.array(n_classes)> hot vector, 'leaf_label': <int>}
        """
        sample = self.ETHEC.__getitem__(item)
        image_folder = sample['image_path'][11:21] + "R" if '.JPG' in sample['image_path'] else sample['image_name'][11:21] + "R"
        path_to_image = os.path.join(self.path_to_images, image_folder,
                                     sample['image_path'] if '.JPG' in sample['image_path'] else sample['image_name'])
        img = cv2.imread(path_to_image)
        if img is None:
            print('This image is None: {} {}'.format(path_to_image, sample['token']))
        ret_sample = {'image': np.array(img, dtype=np.float32),
                      'labels': self.labelmap.get_one_hot(sample['family'], sample['subfamily'], sample['genus'],
                                                          sample['specific_epithet']),
                      'leaf_label': self.labelmap.get_label_id('specific_epithet', sample['specific_epithet'])}

        if self.transform:
            ret_sample = self.transform(ret_sample)
        return ret_sample

    def __len__(self):
        """
        Return number of entries in the database.
        :return: <int> length of database
        """
        return len(self.ETHEC)

    def get_sample(self, token):
        """
        Fetch database entry based on its token.
        :param token: <str> Token used to fetch corresponding entry. (uuid)
        :return: see schema.md
        """
        return self.ETHEC.get_sample(token)


def generate_labelmap(path_to_json):
    """
    Generates entries for labelmap.
    :param path_to_json: <str> Path to .json to read database from.
    :return: -
    """
    ethec = ETHEC(path_to_json)
    family, subfamily, genus, specific_epithet, genus_specific_epithet = {}, {}, {}, {}, {}
    f_c, s_c, g_c, se_c, gse_c = 0, 0, 0, 0, 0
    for sample in tqdm(ethec):
        if sample['family'] not in family:
            family[sample['family']] = f_c
            f_c += 1
        if sample['subfamily'] not in subfamily:
            subfamily[sample['subfamily']] = s_c
            s_c += 1
        if sample['genus'] not in genus:
            genus[sample['genus']] = g_c
            g_c += 1
        if sample['specific_epithet'] not in specific_epithet:
            specific_epithet[sample['specific_epithet']] = se_c
            se_c += 1
        if '{}_{}'.format(sample['genus'], sample['specific_epithet']) not in genus_specific_epithet:
            genus_specific_epithet['{}_{}'.format(sample['genus'], sample['specific_epithet'])] = gse_c
            gse_c += 1
    print(json.dumps(family, indent=4))
    print(json.dumps(subfamily, indent=4))
    print(json.dumps(genus, indent=4))
    print(json.dumps(specific_epithet, indent=4))
    print(json.dumps(genus_specific_epithet, indent=4))


class SplitDataset:
    """
    Splits a given dataset to train, val and test.
    """
    def __init__(self, path_to_json, path_to_images, path_to_save_splits, labelmap, train_ratio=0.8, val_ratio=0.1,
                 test_ratio=0.1):
        """
        Constructor.
        :param path_to_json: <str> Path to .json to read database from.
        :param path_to_images: <str> Path to parent directory that contains the images.
        :param path_to_save_splits: <str> Path to directory where the .json splits are saved.
        :param labelmap: <ETHECLabelMap> Labelmap
        :param train_ratio: <float> Proportion of the dataset used for train.
        :param val_ratio: <float> Proportion of the dataset used for val.
        :param test_ratio: <float> Proportion of the dataset used for test.
        """
        if train_ratio + val_ratio + test_ratio != 1:
            print('Warning: Ratio does not add up to 1.')
        self.path_to_save_splits = path_to_save_splits
        self.path_to_json = path_to_json
        self.database = ETHEC(self.path_to_json)
        self.train_ratio, self.val_ratio, self.test_ratio = train_ratio, val_ratio, test_ratio
        self.labelmap = labelmap
        self.train, self.val, self.test = {}, {}, {}
        self.stats = {}
        self.minimum_samples = 3
        self.minimum_samples_to_use_split = 10
        print('Database has {} sample.'.format(len(self.database)))

    def collect_stats(self):
        """
        Generate counts for each class
        :return: -
        """
        for data_id in range(len(self.database)):
            sample = self.database[data_id]

            label_id = self.labelmap.get_label_id('specific_epithet', sample['specific_epithet'])
            if label_id not in self.stats:
                self.stats[label_id] = [sample['token']]
            else:
                self.stats[label_id].append(sample['token'])
        # print({label_id: len(self.stats[label_id]) for label_id in self.stats})

    def split(self):
        """
        Split data.
        :return: -
        """
        for label_id in self.stats:
            samples_for_label_id = self.stats[label_id]
            n_samples = len(samples_for_label_id)
            if n_samples < self.minimum_samples:
                continue

            # if the number of samples are less than self.minimum_samples_to_use_split then split them equally
            if n_samples < self.minimum_samples_to_use_split:
                n_train_samples, n_val_samples, n_test_samples = n_samples//3, n_samples//3, n_samples//3
            else:
                n_train_samples = int(self.train_ratio * n_samples)
                n_val_samples = int(self.val_ratio * n_samples)
                n_test_samples = int(self.test_ratio * n_samples)

            remaining_samples = n_samples - (n_train_samples + n_val_samples + n_test_samples)
            n_val_samples += remaining_samples % 2 + remaining_samples//2
            n_test_samples += remaining_samples//2

            train_samples_id_list = samples_for_label_id[:n_train_samples]
            val_samples_id_list = samples_for_label_id[n_train_samples:n_train_samples+n_val_samples]
            test_samples_id_list = samples_for_label_id[-n_test_samples:]

            for sample_id in train_samples_id_list:
                self.train[sample_id] = self.database.get_sample(sample_id)
            for sample_id in val_samples_id_list:
                self.val[sample_id] = self.database.get_sample(sample_id)
            for sample_id in test_samples_id_list:
                self.test[sample_id] = self.database.get_sample(sample_id)

    def write_to_disk(self):
        """
        Write the train, val, test .json splits to disk.
        :return: -
        """
        with open(os.path.join(self.path_to_save_splits, 'train.json'), 'w') as fp:
            json.dump(self.train, fp, indent=4)
        with open(os.path.join(self.path_to_save_splits, 'val.json'), 'w') as fp:
            json.dump(self.val, fp, indent=4)
        with open(os.path.join(self.path_to_save_splits, 'test.json'), 'w') as fp:
            json.dump(self.test, fp, indent=4)

    def make_split_to_disk(self):
        """
        Collectively call functions to make splits and save to disk.
        :return: -
        """
        self.collect_stats()
        self.split()
        self.write_to_disk()


class Rescale(object):
    """
    Resize images.
    """
    def __init__(self, output_size):
        """
        Constructor.
        :param output_size: <(int, int)> Tuple specifying the spatial dimensions of the resized image.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """
        Returns sample with resized image.
        :param sample: see ETHECDB
        :return: see ETHECDB
        """
        image, label, leaf_label = sample['image'], sample['labels'], sample['leaf_label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='constant', anti_aliasing=True, anti_aliasing_sigma=None)

        return {'image': img, 'labels': label, 'leaf_label': leaf_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, leaf_label = sample['image'], sample['labels'], sample['leaf_label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'labels': torch.from_numpy(label).float(),
                'leaf_label': leaf_label}


class Normalize(torchvision.transforms.Normalize):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        torchvision.transforms.Normalize.__init__(self, mean, std, inplace)

    def __call__(self, input):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        input['image'] = super(Normalize, self).__call__(input['image'])
        return input


def generate_normalization_values(dataset):
    """
    Calculate mean and std values for a dataset.
    :param dataset: <PyTorch dataset> dataset to calculate mean, std over
    :return: -
    """

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in tqdm(loader):
        batch_samples = data['image'].size(0)
        data = data['image'].view(batch_samples, data['image'].size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print('Mean: {}, Std: {}'.format(mean, std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", help='Parent directory with images.', type=str)
    parser.add_argument("--json_path", help='Path to json with relevant data.', type=str)
    parser.add_argument("--path_to_save_splits", help='Path to json with relevant data.', type=str)
    parser.add_argument("--mode", help='Path to json with relevant data.', type=str)
    args = parser.parse_args()

    labelmap = ETHECLabelMap()
    # mean: tensor([143.2341, 162.8151, 177.2185], dtype=torch.float64)
    # std: tensor([66.7762, 59.2524, 51.5077], dtype=torch.float64)

    if args.mode == 'split':
        # create files with train, val and test splits
        data_splitter = SplitDataset(args.json_path, args.images_dir, args.path_to_save_splits, ETHECLabelMap())
        data_splitter.make_split_to_disk()

    elif args.mode == 'calc_mean_std':
        tform = transforms.Compose([Rescale((224, 224)),
                                    ToTensor()])
        train_set = ETHECDB(path_to_json='../database/ETHEC/train.json',
                            path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                            labelmap=labelmap, transform=tform)
        generate_normalization_values(train_set)

    else:
        tform = transforms.Compose([Rescale((224, 224)),
                                    ToTensor(),
                                    Normalize(mean=(143.2341, 162.8151, 177.2185), std=(66.7762, 59.2524, 51.5077))])
        train_set = ETHECDB(path_to_json='../database/ETHEC/train.json',
                            path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                            labelmap=labelmap, transform=tform)
        val_set = ETHECDB(path_to_json='../database/ETHEC/val.json',
                          path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                          labelmap=labelmap, transform=tform)
        test_set = ETHECDB(path_to_json='../database/ETHEC/test.json',
                           path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                           labelmap=labelmap, transform=tform)
        print('Dataset has following splits: train: {}, val: {}, test: {}'.format(len(train_set), len(val_set),
                                                                                  len(test_set)))
        print(train_set[0]['image'])
