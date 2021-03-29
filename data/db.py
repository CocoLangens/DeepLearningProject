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
from PIL import Image
import numpy as np
import random


class ETHECLabelMap:
    """
    Implements map from labels to hot vectors for ETHEC database.
    """

    def __init__(self):
        self.family = {
            "vertebrate.n.01": 0,
            "invertebrate.n.01": 1
        }
        self.subfamily = {
            "aquatic_vertebrate.n.01": 0,
            "bird.n.01": 1,
            "amphibian.n.03": 2,
            "reptile.n.01": 3,
            "mammal.n.01": 4,
            "arthropod.n.01": 5,
            "echinoderm.n.01": 6,
            "coelenterate.n.01": 7,
            "mollusk.n.01": 8
        }

        self.genus = {
            "bony_fish.n.01": 0,
            "seabird.n.01": 1,
            "waterfowl.n.01": 2,
            "wading_bird.n.01": 3,
            "frog.n.01": 4,
            "salamader.n.01": 5,
            "snake.n.01": 6,
            "crocodillian_reptile.n.01": 7,
            "ungulate.n.01": 8,
            "carnivore.n.01": 9,
            "primate.n.02": 10,
            "aquatic_mammal.n.01": 11,
            "rodent.n.01": 12,
            "pachyderm.n.01": 13,
            "marsupial.n.01": 14,
            "trilobite.n.01": 15,
            "arachnid.n.01": 16,
            "crustacean.n.01": 17,
            "centipede.n.01": 18,
            "insect.n.01": 19,
            "sea cucumber.n.01": 20,
            "jellyfish.n.02": 21,
            "anthozoan.n.01": 22,
            "gastropod.n.01": 23
        }

        self.specific_epithet = {
            "goldfish.n.01": 0,
            "european_fire_salamander.n.01": 1,
            "bullfrog.n.01": 2,
            "tailed_frog.n.01": 3,
            "american_alligator.n.01": 4,
            "boa_constrictor.n.01": 5,
            "trilobite.n.01": 6,
            "scorpion.n.03": 7,
            "black_widow.n.01": 8,
            "tarantula.n.02": 9,
            "centipede.n.01": 10,
            "goose.n.01": 11,
            "koala.n.01": 12,
            "jellyfish.n.02": 13,
            "brain_coral.n.01": 14,
            "snail.n.01": 15,
            "slug.n.07": 16,
            "sea_slug.n.01": 17,
            "american_lobster.n.02": 18,
            "spiny_lobster.n.02": 19,
            "black_stork.n.01": 20,
            "king_penguin.n.01": 21,
            "albatross.n.02": 22,
            "dugong.n.01": 23,
            "chihuahua.n.03": 24,
            "yorkshire_terrier.n.01": 25,
            "golden_retriever.n.01": 26,
            "labrador_retriever.n.01": 27,
            "german_shepherd.n.01": 28,
            "standard_poodle.n.01": 29,
            "tabby.n.01": 30,
            "persian_cat.n.01": 31,
            "egyptian_cat.n.01": 32,
            "cougar.n.01": 33,
            "loin.n.01": 34,
            "brown_bear.n.01": 35,
            "ladybug.n.01": 36,
            "fly.n.01": 37,
            "bee.n.01": 38,
            "grasshopper.n.01": 39,
            "walking_stick.n.02": 40,
            "cockroach.n.01": 41,
            "mantis.n.01": 42,
            "dragonfly.n.01": 43,
            "monarch.n.02": 44,
            "sulphur_butterfly.n.01": 45,
            "sea_cucumber.n.01": 46,
            "guinea_pig.n.02": 47,
            "hog.n.03": 48,
            "ox.n.01": 49,
            "bison.n.01": 50,
            "bighorn.n.02": 51,
            "gazelle.n.01": 52,
            "arabian_camel.n.01": 53,
            "orangutan.n.01": 54,
            "chimpanzee.n.01": 55,
            "baboon.n.01": 56,
            "african_elephant.n.01": 57,
            "lesser_panda.n.01": 58
        }

        self.genus_specific_epithet = {
            "goldfish.n.01": 0,
            "european_fire_salamander.n.01": 1,
            "bullfrog.n.01": 2,
            "tailed_frog.n.01": 3,
            "american_alligator.n.01": 4,
            "boa_constrictor.n.01": 5,
            "trilobite.n.01": 6,
            "scorpion.n.03": 7,
            "black_widow.n.01": 8,
            "tarantula.n.02": 9,
            "centipede.n.01": 10,
            "goose.n.01": 11,
            "koala.n.01": 12,
            "jellyfish.n.02": 13,
            "brain_coral.n.01": 14,
            "snail.n.01": 15,
            "slug.n.07": 16,
            "sea_slug.n.01": 17,
            "american_lobster.n.02": 18,
            "spiny_lobster.n.02": 19,
            "black_stork.n.01": 20,
            "king_penguin.n.01": 21,
            "albatross.n.02": 22,
            "dugong.n.01": 23,
            "chihuahua.n.03": 24,
            "yorkshire_terrier.n.01": 25,
            "golden_retriever.n.01": 26,
            "labrador_retriever.n.01": 27,
            "german_shepherd.n.01": 28,
            "standard_poodle.n.01": 29,
            "tabby.n.01": 30,
            "persian_cat.n.01": 31,
            "egyptian_cat.n.01": 32,
            "cougar.n.01": 33,
            "loin.n.01": 34,
            "brown_bear.n.01": 35,
            "ladybug.n.01": 36,
            "fly.n.01": 37,
            "bee.n.01": 38,
            "grasshopper.n.01": 39,
            "walking_stick.n.02": 40,
            "cockroach.n.01": 41,
            "mantis.n.01": 42,
            "dragonfly.n.01": 43,
            "monarch.n.02": 44,
            "sulphur_butterfly.n.01": 45,
            "sea_cucumber.n.01": 46,
            "guinea_pig.n.02": 47,
            "hog.n.03": 48,
            "ox.n.01": 49,
            "bison.n.01": 50,
            "bighorn.n.02": 51,
            "gazelle.n.01": 52,
            "arabian_camel.n.01": 53,
            "orangutan.n.01": 54,
            "chimpanzee.n.01": 55,
            "baboon.n.01": 56,
            "african_elephant.n.01": 57,
            "lesser_panda.n.01": 58
        }

        self.child_of_family = {
            "vertebrate.n.01": [
                "aquatic_vertebrate.n.01",
                "bird.n.01",
                "amphibian.n.03",
                "reptile.n.01",
                "mammal.n.01"
            ],
            "invertebrate.n.01": [
                "arthropod.n.01",
                "echinoderm.n.01",
                "coelenterate.n.01",
                "mollusk.n.01"
            ]
        }

        self.child_of_subfamily = {
            "aquatic_vertebrate.n.01": [
                "bony_fish.n.01"
            ],
            "bird.n.01": [
                "seabird.n.01",
                "waterfowl.n.01",
                "wading_bird.n.01"
            ],
            "amphibian.n.03": [
                "frog.n.01",
                "salamander.n.01"
            ],
            "reptile.n.01": [
                "snake.n.01",
                "crocodillian_reptile.n.01"
            ],
            "mammal.n.01": [
                "ungulat.n.01",
                "carnivore.n.01",
                "primate.n.02",
                "aquatic_mammal.n.01",
                "rodent.n.01",
                "pachyderm.n.01",
                "marsupial.n.01"
            ],
            "arthropod.n.01": [
                "trilobite.n.01",
                "arachnid.n.01",
                "crustacean.n.01",
                "centipede.n.01",
                "insect.n.01"
            ],
            "echinoderm.n.01": [
                "sea_cucumber.n.01"
            ],
            "coelenterate.n.01": [
                "anthozoan.n.01",
                "jellyfish.n.02"
            ],
            "mollusk.n.01": [
                "gastropod.n.01"
            ]
        }

        self.child_of_genus = {
            "bony_fish.n.01": [
                "goldfish.n.01"
            ],
            "salamander.n.01": [
                "european_fire_salamander.n.01"
            ],
            "frog.n.01": [
                "bullfrog.n.01",
                "tailed_frog.n.01"
            ],
            "snake.n.01": [
                "boa_constrictor.n.01"
            ],
            "crocodillian_reptile.n.01": [
                "american_alligator.n.01"
            ],
            "ungulate.n.01": [
                "ox.n.01",
                "bison.n.01",
                "bighorn.n.02",
                "gazelle.n.01",
                "arabian_camel.n.01",
                "hog.n.03"
            ],
            "carnivore.n.01": [
                "chihuahua.n.03",
                "yorkshire_terrier.n.01",
                "golden_retriever.n.01",
                "labrador_retriever.n.01",
                "german_shepherd.n.01",
                "standard_poodle.n.01",
                "tabby.n.01",
                "persian_cat.n.01",
                "egyptian_cat.n.01",
                "cougar.n.01",
                "lion.n.01",
                "brown_bear.n.01",
                "lesser_panda.n.01"
            ],
            "primate.n.02": [
                "baboon.n.01",
                "orangutan.n.01",
                "chimpanzee.n.01"
            ],
            "aquatic_mammal.n.01": [
                "dugong.n.01"
            ],
            "rodent.n.01": [
                "guinea_pig.n.02"
            ],
            "pachyderm.n.01": [
                "african_elephant.n.01"
            ],
            "marsupial.n.01": [
                "koala.n.01"
            ],
            "seabird.n.01": [
                "king_penguin.n.01",
                "albatross.n.02"
            ],
            "waterfowl.n.01": [
                "goose.n.01"
            ],
            "wading_bird.n.01": [
                "black_stork.n.01"
            ],
            "trilobite.n.01": [
                "trilobite.n.01"
            ],
            "arachnid.n.01": [
                "black_widow.n.01",
                "tarantula.n.02",
                "scorpion.n.03"
            ],
            "crustacean.n.01": [
                "american_lobster.n.02",
                "spiny_lobster.n.02"
            ],
            "centipede.n.01": [
                "centipede.n.01"
            ],
            "insect.n.01": [
                "ladybug.n.01",
                "fly.n.01",
                "bee.n.01",
                "grasshopper.n.01",
                "walking_stick.n.02",
                "cockroach.n.01",
                "mantis.n.01",
                "dragonfly.n.01",
                "monarch.n.02",
                "sulphur_butterfly.n.01"
            ],
            "sea_cucumber.n.01": [
                "sea_cucumber.n.01"
            ],
            "anthozoan.n.01": [
                "brain_coral.n.01"
            ],
            "jellyfish.n.02": [
                "jellyfish.n.02"
            ],
            "gastropod.n.01": [
                "snail.n.01",
                "slug.n.07",
                "sea_slug.n.01"
            ]
        }

        self.levels = [len(self.family), len(self.subfamily), len(self.genus), len(self.genus_specific_epithet)]
        self.n_classes = sum(self.levels)
        self.classes = [key for class_list in [self.family, self.subfamily, self.genus, self.genus_specific_epithet] for key
                        in class_list]
        self.level_names = ['family', 'subfamily', 'genus', 'genus_specific_epithet']

        self.convert_child_of()

    def convert_child_of(self):
        self.level_stop, self.level_start = [], []
        for level_id, level_len in enumerate(self.levels):
            if level_id == 0:
                self.level_start.append(0)
                self.level_stop.append(level_len)
            else:
                self.level_start.append(self.level_stop[level_id - 1])
                self.level_stop.append(self.level_stop[level_id - 1] + level_len)

        self.child_of_family_ix, self.child_of_subfamily_ix, self.child_of_genus_ix = {}, {}, {}
        for family_name in self.child_of_family:
            if family_name not in self.family:
                continue
            self.child_of_family_ix[self.family[family_name]] = []
            for subfamily_name in self.child_of_family[family_name]:
                if subfamily_name not in self.subfamily:
                    continue
                self.child_of_family_ix[self.family[family_name]].append(self.subfamily[subfamily_name])

        for subfamily_name in self.child_of_subfamily:
            if subfamily_name not in self.subfamily:
                continue
            self.child_of_subfamily_ix[self.subfamily[subfamily_name]] = []
            for genus_name in self.child_of_subfamily[subfamily_name]:
                if genus_name not in self.genus:
                    continue
                self.child_of_subfamily_ix[self.subfamily[subfamily_name]].append(self.genus[genus_name])

        for genus_name in self.child_of_genus:
            if genus_name not in self.genus:
                continue
            self.child_of_genus_ix[self.genus[genus_name]] = []
            for genus_specific_epithet_name in self.child_of_genus[genus_name]:
                if genus_specific_epithet_name not in self.genus_specific_epithet:
                    continue
                self.child_of_genus_ix[self.genus[genus_name]].append(self.genus_specific_epithet[genus_specific_epithet_name])

        self.family_ix_to_str = {self.family[k]: k for k in self.family}
        self.subfamily_ix_to_str = {self.subfamily[k]: k for k in self.subfamily}
        self.genus_ix_to_str = {self.genus[k]: k for k in self.genus}
        self.genus_specific_epithet_ix_to_str = {self.genus_specific_epithet[k]: k for k in self.genus_specific_epithet}

    def get_one_hot(self, family, subfamily, genus, specific_epithet):
        retval = np.zeros(self.n_classes)
        retval[self.family[family]] = 1
        retval[self.subfamily[subfamily] + self.levels[0]] = 1
        retval[self.genus[genus] + self.levels[0] + self.levels[1]] = 1
        retval[self.specific_epithet[specific_epithet] + self.levels[0] + self.levels[1] + self.levels[2]] = 1
        return retval

    def get_label_id(self, level_name, label_name):
        return getattr(self, level_name)[label_name]

    def get_level_labels(self, family, subfamily, genus, specific_epithet):
        return np.array([
            self.get_label_id('family', family),
            self.get_label_id('subfamily', subfamily),
            self.get_label_id('genus', genus),
            self.get_label_id('specific_epithet', specific_epithet)
        ])

    def get_children_of(self, c_ix, level_id):
        if level_id == 0:
            # possible family
            return [self.family[k] for k in self.family]
        elif level_id == 1:
            # possible_subfamily
            return self.child_of_family_ix[c_ix]
        elif level_id == 2:
            # possible_genus
            return self.child_of_subfamily_ix[c_ix]
        elif level_id == 3:
            # possible_genus_specific_epithet
            return self.child_of_genus_ix[c_ix]
        else:
            return None

    def decode_children(self, level_labels):
        level_labels = level_labels.cpu().numpy()
        possible_family = [self.family[k] for k in self.family]
        possible_subfamily = self.child_of_family_ix[level_labels[0]]
        possible_genus = self.child_of_subfamily_ix[level_labels[1]]
        possible_genus_specific_epithet = self.child_of_genus_ix[level_labels[2]]
        new_level_labels = [
            level_labels[0],
            possible_subfamily.index(level_labels[1]),
            possible_genus.index(level_labels[2]),
            possible_genus_specific_epithet.index(level_labels[3])
        ]
        return {'family': possible_family, 'subfamily': possible_subfamily, 'genus': possible_genus,
                'genus_specific_epithet': possible_genus_specific_epithet}, new_level_labels


class ETHECLabelMapMerged(ETHECLabelMap):
    def __init__(self):
        ETHECLabelMap.__init__(self)
        self.levels = [len(self.family), len(self.subfamily), len(self.genus), len(self.genus_specific_epithet)]
        self.n_classes = sum(self.levels)
        self.classes = [key for class_list in [self.family, self.subfamily, self.genus, self.genus_specific_epithet] for
                        key
                        in class_list]
        self.level_names = ['family', 'subfamily', 'genus', 'genus_specific_epithet']
        self.convert_child_of()

    def get_one_hot(self, family, subfamily, genus, genus_specific_epithet):
        retval = np.zeros(self.n_classes)
        retval[self.family[family]] = 1
        retval[self.subfamily[subfamily] + self.levels[0]] = 1
        retval[self.genus[genus] + self.levels[0] + self.levels[1]] = 1
        retval[
            self.genus_specific_epithet[genus_specific_epithet] + self.levels[0] + self.levels[1] + self.levels[2]] = 1
        return retval

    def get_label_id(self, level_name, label_name):
        return getattr(self, level_name)[label_name]

    def get_level_labels(self, family, subfamily, genus, genus_specific_epithet):
        return np.array([
            self.get_label_id('family', family),
            self.get_label_id('subfamily', subfamily),
            self.get_label_id('genus', genus),
            self.get_label_id('genus_specific_epithet', genus_specific_epithet)
        ])


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


class ETHECSmall(ETHEC):
    """
    ETHEC iterator.
    """

    def __init__(self, path_to_json, single_level=False):
        """
        Constructor.
        :param path_to_json: <str> .json path used for loading database entries.
        """
        lmap = ETHECLabelMapMergedSmall(single_level)
        self.path_to_json = path_to_json
        with open(path_to_json) as json_file:
            self.data_dict = json.load(json_file)
        # print([token for token in self.data_dict])
        if single_level:
            self.data_tokens = [token for token in self.data_dict
                                if self.data_dict[token]['family'] in lmap.family]
        else:
            self.data_tokens = [token for token in self.data_dict
                                if '{}_{}'.format(self.data_dict[token]['genus'],
                                                  self.data_dict[token]['specific_epithet'])
                                in lmap.genus_specific_epithet]


class ETHECLabelMapMergedSmall(ETHECLabelMapMerged):
    def __init__(self, single_level=False):
        self.single_level = single_level
        ETHECLabelMapMerged.__init__(self)

        self.family = {
            # "dummy1": 0,
            "Hesperiidae": 0,
            "Riodinidae": 1,
            "Lycaenidae": 2,
            "Papilionidae": 3,
            "Pieridae": 4
        }
        if self.single_level:
            print('== Using single_level data')
            self.levels = [len(self.family)]
            self.n_classes = sum(self.levels)
            self.classes = [key for class_list in [self.family] for key
                            in class_list]
            self.level_names = ['family']
        else:
            self.subfamily = {
                "Hesperiinae": 0,
                "Pyrginae": 1,
                "Nemeobiinae": 2,
                "Polyommatinae": 3,
                "Parnassiinae": 4,
                "Pierinae": 5
            }
            self.genus = {
                "Ochlodes": 0,
                "Hesperia": 1,
                "Pyrgus": 2,
                "Spialia": 3,
                "Hamearis": 4,
                "Polycaena": 5,
                "Agriades": 6,
                "Parnassius": 7,
                "Aporia": 8
            }
            self.genus_specific_epithet = {
                "Ochlodes_venata": 0,
                "Hesperia_comma": 1,
                "Pyrgus_alveus": 2,
                "Spialia_sertorius": 3,
                "Hamearis_lucina": 4,
                "Polycaena_tamerlana": 5,
                "Agriades_lehanus": 6,
                "Parnassius_jacquemonti": 7,
                "Aporia_crataegi": 8,
                "Aporia_procris": 9,
                "Aporia_potanini": 10,
                "Aporia_nabellica": 11

            }
            self.levels = [len(self.family), len(self.subfamily), len(self.genus), len(self.genus_specific_epithet)]
            self.n_classes = sum(self.levels)
            self.classes = [key for class_list in [self.family, self.subfamily, self.genus, self.genus_specific_epithet]
                            for key in class_list]
            self.level_names = ['family', 'subfamily', 'genus', 'genus_specific_epithet']
            self.convert_child_of()

    def get_one_hot(self, family, subfamily, genus, genus_specific_epithet):
        retval = np.zeros(self.n_classes)
        retval[self.family[family]] = 1
        if not self.single_level:
            retval[self.subfamily[subfamily] + self.levels[0]] = 1
            retval[self.genus[genus] + self.levels[0] + self.levels[1]] = 1
            retval[self.genus_specific_epithet[genus_specific_epithet] + self.levels[0] + self.levels[1] + self.levels[
                2]] = 1
        return retval

    def get_label_id(self, level_name, label_name):
        return getattr(self, level_name)[label_name]

    def get_level_labels(self, family, subfamily=None, genus=None, genus_specific_epithet=None):
        if not self.single_level:
            return np.array([
                self.get_label_id('family', family),
                self.get_label_id('subfamily', subfamily),
                self.get_label_id('genus', genus),
                self.get_label_id('genus_specific_epithet', genus_specific_epithet)
            ])
        else:
            return np.array([
                self.get_label_id('family', family)
            ])


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
        #image_folder = sample['image_path'][11:21] + "R" if '.JPG' in sample['image_path'] else sample['image_name'][
                                                                                                #11:21] + "R"
    
        image_folder = sample['image_name']#[11:21] 
        print(image_folder, "image_older")
        path_to_image = os.path.join(self.path_to_images,sample['image_path'],sample['image_name'])
                            
        
        #path_to_image = os.path.join(self.path_to_images, image_folder,
                                     #sample['image_path'] if '.JPG' in sample['image_path'] else sample['image_name'])
        img = cv2.imread(path_to_image)
        #Aangepast    
        print(type(img), "type")
        #untill here
        if img is None:
            print('This image is None: {} {}'.format(path_to_image, sample['token']))

        if self.transform:
            img = self.transform(img)

        ret_sample = {
            'image': img,
            'labels': torch.from_numpy(self.labelmap.get_one_hot(sample['family'], sample['subfamily'], sample['genus'],
                                                                 sample['specific_epithet'])).float(),
            'leaf_label': self.labelmap.get_label_id('specific_epithet', sample['specific_epithet']),
            'level_labels': torch.from_numpy(self.labelmap.get_level_labels(sample['family'], sample['subfamily'],
                                                                            sample['genus'],
                                                                            sample['specific_epithet'])).long(),
            'path_to_image': path_to_image
        }
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


class ETHECDBMerged(ETHECDB):
    """
    Creates a PyTorch dataset.
    """

    def __init__(self, path_to_json, path_to_images, labelmap, transform=None, with_images=True):
        """
        Constructor.
        :param path_to_json: <str> Path to .json from which to read database entries.
        :param path_to_images: <str> Path to parent directory where images are stored.
        :param labelmap: <ETHECLabelMap> Labelmap.
        :param transform: <torchvision.transforms> Set of transforms to be applied to the entries in the database.
        """
        ETHECDB.__init__(self, path_to_json, path_to_images, labelmap, transform)
        self.with_images = with_images

    def __getitem__(self, item):
        """
        Fetch an entry based on index.
        :param item: <int> Index to fetch.
        :return: <dict> Consumable object (see schema.md)
                {'image': <np.array> image, 'labels': <np.array(n_classes)> hot vector, 'leaf_label': <int>}
        """

        sample = self.ETHEC.__getitem__(item)
        if self.with_images:
            image_folder = sample['image_path'][11:21] + "R" if '.JPG' in sample['image_path'] else sample['image_name'][
                                                                                                    11:21] + "R"
            path_to_image = os.path.join(self.path_to_images, image_folder,
                                         sample['image_path'] if '.JPG' in sample['image_path'] else sample['image_name'])
            img = cv2.imread(path_to_image)
            if img is None:
                print('This image is None: {} {}'.format(path_to_image, sample['token']))

            img = np.array(img)
            if self.transform:
                img = self.transform(img)
        else:
            image_folder = sample['image_path'][11:21] + "R" if '.JPG' in sample['image_path'] else sample[
                                                                                                        'image_name'][
                                                                                                    11:21] + "R"
            path_to_image = os.path.join(self.path_to_images, image_folder,
                                         sample['image_path'] if '.JPG' in sample['image_path'] else sample[
                                             'image_name'])
            img = 0

        ret_sample = {
            'image': img,
            'image_filename': sample['image_path'] if '.JPG' in sample['image_path'] else sample['image_name'],
            'labels': torch.from_numpy(self.labelmap.get_one_hot(sample['family'], sample['subfamily'], sample['genus'],
                                                                 '{}_{}'.format(sample['genus'],
                                                                                sample['specific_epithet']))).float(),
            'leaf_label': self.labelmap.get_label_id('genus_specific_epithet',
                                                     '{}_{}'.format(sample['genus'], sample['specific_epithet'])),
            'level_labels': torch.from_numpy(self.labelmap.get_level_labels(sample['family'], sample['subfamily'],
                                                                            sample['genus'],
                                                                            '{}_{}'.format(sample['genus'], sample[
                                                                                'specific_epithet']))).long(),
            'path_to_image': path_to_image
        }
        return ret_sample


class ETHECDBMergedSmall(ETHECDBMerged):
    """
    Creates a PyTorch dataset.
    """

    def __init__(self, path_to_json, path_to_images, labelmap, transform=None, with_images=True):
        """
        Constructor.
        :param path_to_json: <str> Path to .json from which to read database entries.
        :param path_to_images: <str> Path to parent directory where images are stored.
        :param labelmap: <ETHECLabelMap> Labelmap.
        :param transform: <torchvision.transforms> Set of transforms to be applied to the entries in the database.
        """
        ETHECDBMerged.__init__(self, path_to_json, path_to_images, labelmap, transform, with_images)
        if hasattr(labelmap, 'single_level'):
            self.ETHEC = ETHECSmall(self.path_to_json, labelmap.single_level)
        else:
            self.ETHEC = ETHECSmall(self.path_to_json)


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

            label_id = self.labelmap.get_label_id('genus_specific_epithet',
                                                  '{}_{}'.format(sample['genus'], sample['specific_epithet']))
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
                n_train_samples, n_val_samples, n_test_samples = n_samples // 3, n_samples // 3, n_samples // 3
            else:
                n_train_samples = int(self.train_ratio * n_samples)
                n_val_samples = int(self.val_ratio * n_samples)
                n_test_samples = int(self.test_ratio * n_samples)

            remaining_samples = n_samples - (n_train_samples + n_val_samples + n_test_samples)
            n_val_samples += remaining_samples % 2 + remaining_samples // 2
            n_test_samples += remaining_samples // 2

            # print(label_id, n_train_samples, n_val_samples, n_test_samples)

            train_samples_id_list = samples_for_label_id[:n_train_samples]
            val_samples_id_list = samples_for_label_id[n_train_samples:n_train_samples + n_val_samples]
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
        with open(os.path.join(self.path_to_save_splits, 'train_merged.json'), 'w') as fp:
            json.dump(self.train, fp, indent=4)
        with open(os.path.join(self.path_to_save_splits, 'val_merged.json'), 'w') as fp:
            json.dump(self.val, fp, indent=4)
        with open(os.path.join(self.path_to_save_splits, 'test_merged.json'), 'w') as fp:
            json.dump(self.test, fp, indent=4)

    def make_split_to_disk(self):
        """
        Collectively call functions to make splits and save to disk.
        :return: -
        """
        self.collect_stats()
        self.split()
        self.write_to_disk()


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


def print_labelmap():
    path_to_json = '../database/ETHEC/'
    with open(os.path.join(path_to_json, 'train.json')) as json_file:
        data_dict = json.load(json_file)
    family, subfamily, genus, specific_epithet, genus_specific_epithet = {}, {}, {}, {}, {}
    f_c, sf_c, g_c, se_c, gse_c = 0, 0, 0, 0, 0
    # to store the children for each node
    child_of_family, child_of_subfamily, child_of_genus = {}, {}, {}
    for key in data_dict:
        if data_dict[key]['family'] not in family:
            family[data_dict[key]['family']] = f_c
            child_of_family[data_dict[key]['family']] = []
            f_c += 1
        if data_dict[key]['subfamily'] not in subfamily:
            subfamily[data_dict[key]['subfamily']] = sf_c
            child_of_subfamily[data_dict[key]['subfamily']] = []
            child_of_family[data_dict[key]['family']].append(data_dict[key]['subfamily'])
            sf_c += 1
        if data_dict[key]['genus'] not in genus:
            genus[data_dict[key]['genus']] = g_c
            child_of_genus[data_dict[key]['genus']] = []
            child_of_subfamily[data_dict[key]['subfamily']].append(data_dict[key]['genus'])
            g_c += 1
        if data_dict[key]['specific_epithet'] not in specific_epithet:
            specific_epithet[data_dict[key]['specific_epithet']] = se_c
            se_c += 1
        if '{}_{}'.format(data_dict[key]['genus'], data_dict[key]['specific_epithet']) not in genus_specific_epithet:
            genus_specific_epithet['{}_{}'.format(data_dict[key]['genus'], data_dict[key]['specific_epithet'])] = gse_c
            specific_epithet[data_dict[key]['specific_epithet']] = se_c
            child_of_genus[data_dict[key]['genus']].append(
                '{}_{}'.format(data_dict[key]['genus'], data_dict[key]['specific_epithet']))
            gse_c += 1
    print(json.dumps(family, indent=4))
    print(json.dumps(subfamily, indent=4))
    print(json.dumps(genus, indent=4))
    print(json.dumps(specific_epithet, indent=4))
    print(json.dumps(genus_specific_epithet, indent=4))

    print(json.dumps(child_of_family, indent=4))
    print(json.dumps(child_of_subfamily, indent=4))
    print(json.dumps(child_of_genus, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", help='Parent directory with images.', type=str)
    parser.add_argument("--json_path", help='Path to json with relevant data.', type=str)
    parser.add_argument("--path_to_save_splits", help='Path to json with relevant data.', type=str)
    parser.add_argument("--mode", help='Path to json with relevant data. [split, calc_mean_std, small]', type=str)
    args = parser.parse_args()

    labelmap = ETHECLabelMap()
    # mean: tensor([143.2341, 162.8151, 177.2185], dtype=torch.float64)
    # std: tensor([66.7762, 59.2524, 51.5077], dtype=torch.float64)

    if args.mode == 'split':
        # create files with train, val and test splits
        data_splitter = SplitDataset(args.json_path, args.images_dir, args.path_to_save_splits, ETHECLabelMapMerged())
        data_splitter.make_split_to_disk()

    elif args.mode == 'show_labelmap':
        print_labelmap()

    elif args.mode == 'calc_mean_std':
        tform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
        train_set = ETHECDB(path_to_json='../database/ETHEC/train.json',
                            path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                            labelmap=labelmap, transform=tform)
        generate_normalization_values(train_set)
    elif args.mode == 'small':
        labelmap = ETHECLabelMapMergedSmall(single_level=True)
        initial_crop = 324
        input_size = 224
        train_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize((initial_crop, initial_crop)),
                                                    transforms.RandomCrop((input_size, input_size)),
                                                    transforms.RandomHorizontalFlip(),
                                                    # ColorJitter(brightness=0.2, contrast=0.2),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                                         std=(66.7762, 59.2524, 51.5077))])
        val_test_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.Resize((input_size, input_size)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                                            std=(66.7762, 59.2524, 51.5077))])
        train_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/train.json',
                                       path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                                       labelmap=labelmap, transform=train_data_transforms)
        val_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/val.json',
                                     path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                                     labelmap=labelmap, transform=val_test_data_transforms)
        test_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/test.json',
                                      path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                                      labelmap=labelmap, transform=val_test_data_transforms)
        print('Dataset has following splits: train: {}, val: {}, test: {}'.format(len(train_set), len(val_set),
                                                                                  len(test_set)))
        print(train_set[0])
    else:
        labelmap = ETHECLabelMapMerged()
        initial_crop = 324
        input_size = 224
        train_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize((initial_crop, initial_crop)),
                                                    transforms.RandomCrop((input_size, input_size)),
                                                    transforms.RandomHorizontalFlip(),
                                                    # ColorJitter(brightness=0.2, contrast=0.2),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                                         std=(66.7762, 59.2524, 51.5077))])
        val_test_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.Resize((input_size, input_size)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                                            std=(66.7762, 59.2524, 51.5077))])
        train_set = ETHECDBMerged(path_to_json='../database/ETHEC/train.json',
                                  path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                                  labelmap=labelmap, transform=train_data_transforms)
        val_set = ETHECDBMerged(path_to_json='../database/ETHEC/val.json',
                                path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                                labelmap=labelmap, transform=val_test_data_transforms)
        test_set = ETHECDBMerged(path_to_json='../database/ETHEC/test.json',
                                 path_to_images='/media/ankit/DataPartition/IMAGO_build/',
                                 labelmap=labelmap, transform=val_test_data_transforms)
        print('Dataset has following splits: train: {}, val: {}, test: {}'.format(len(train_set), len(val_set),
                                                                                  len(test_set)))
        print(train_set[0])
