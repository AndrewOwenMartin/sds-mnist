# Copyright 2020 Andrew Owen Martin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections, datetime, functools
import json, logging, random
import sds
import mnist_loader
import sys

log = logging.getLogger(__name__)


class Picture:
    """ An interface to a MNIST dataset image pixel array """

    def __init__(self, width=None, height=None, pixels=None, label=None, index=None):

        if pixels is None and width and height:

            pixels = []

            for row_num in range(height):

                row = []

                for col_num in range(width):

                    row.append(rng.randrange(width * height))

                pixels.append(row)

        elif pixels and width is None and height is None:

            height = len(pixels)
            width = len(pixels[0])

            pass

        else:

            raise ValueError("Either pixels or width and height")

        self.pixels = pixels
        self.width = width
        self.height = height
        self.label = label
        self.index = index

    def brightness(self, pos):

        return self.pixels[pos[1]][pos[0]]

    def __str__(self):

        text_col_width = len(str(self.width * self.height))

        return (
            "\n".join(
                " ".join(f"{col:{text_col_width}}" for col in row)
                for row in self.pixels
            )
            + f"\nLabel: {self.label}"
        )


class PictureMicrotestList(collections.UserList):
    """ An interface to all the possible Minchinton tests one could perform on
    an image """

    def minchinton(img, a, b):

        a_lvl = img.brightness(a)
        b_lvl = img.brightness(b)

        diff = b_lvl - a_lvl

        log.debug("a: %s = %s, b: %s = %s. Diff = %s", a, a_lvl, b, b_lvl, diff)

        return min(max(diff, -1), 1)

    def microtest(hyp_image, a, b, model_image):

        hyp_minchinton = PictureMicrotestList.minchinton(hyp_image, a, b)
        model_minchinton = PictureMicrotestList.minchinton(model_image, a, b)

        return hyp_minchinton == model_minchinton

    def __init__(self, model_image):

        self.width = model_image.width
        self.height = model_image.height

        self.model_image = model_image

        self.microtest = functools.partial(
            PictureMicrotestList.microtest, model_image=self.model_image
        )

    def __len__(self):

        return pow(self.width * self.height, 2)

    def __getitem__(self, key):

        y2 = key % self.height
        x2 = key // self.height % self.width
        y1 = key // (self.height * self.width) % self.height
        x1 = key // (self.height * self.width * self.height) % self.width
        a = (x1, y1)
        b = (x2, y2)

        return functools.partial(self.microtest, a=a, b=b)


def get_hypotheses_and_microtests(rng, max_items=None):
    """ Loads images from the MNIST dataset """

    test_pictures = [
        Picture(pixels=pixels, label=label, index=num)
        for num, (pixels, label) in enumerate(
            zip(*mnist_loader.get_10k_data(max_items=max_items))
        )
    ]

    training_pictures = [
        Picture(pixels=pixels, label=label, index=num)
        for num, (pixels, label) in enumerate(
            zip(*mnist_loader.get_60k_data(max_items=max_items))
        )
    ]

    hypotheses = list(training_pictures)

    model_image = rng.choice(test_pictures)

    microtests = PictureMicrotestList(model_image=model_image)

    return test_pictures, hypotheses, microtests


def example(agent_count, max_iterations, rng):

    test_pictures, hypotheses, microtests = get_hypotheses_and_microtests(
        max_items=100, rng=rng
    )

    log.info("Selected this picture\n%s", microtests.model_image)

    swarm = sds.Swarm(agent_count=agent_count)

    log.info(
        "Doing standard SDS with %s agents for %s iterations.",
        agent_count,
        max_iterations,
    )

    standard_sds(
        hypotheses=hypotheses,
        microtests=microtests,
        max_iterations=max_iterations,
        swarm=swarm,
        rng=rng,
    )

    guess = swarm.largest_cluster.hyp.label

    is_correct = guess == microtests.model_image.label

    log.info("Guess: %s, Correct: %s", guess, is_correct)


def standard_sds(hypotheses, microtests, swarm, max_iterations, rng):

    DH = sds.DH_uniform(hypotheses=hypotheses, rng=rng)

    D = sds.D_passive(DH=DH, swarm=swarm, rng=rng)

    TM = sds.TM_uniform(microtests, rng=rng)

    T = sds.T_boolean(TM=TM)

    I = sds.I_sync(D=D, T=T, swarm=swarm)

    H = sds.H_fixed(max_iterations)

    sds.SDS(I=I, H=H)


def experiment(rng, agent_count=200, max_iterations=200):

    start = datetime.datetime.now()

    test_pictures, hypotheses, microtests = get_hypotheses_and_microtests(
        max_items=None, rng=rng
    )

    correct = 0

    results = []

    path = "results/standard_sds_results.json"

    for num, test_picture in enumerate(test_pictures):

        microtests = PictureMicrotestList(model_image=test_picture)

        swarm = sds.Swarm(agent_count=agent_count)

        standard_sds(
            hypotheses=hypotheses,
            microtests=microtests,
            swarm=swarm,
            max_iterations=max_iterations,
            rng=rng,
        )

        hyp = swarm.largest_cluster.hyp

        guess = hyp.label

        guess_num = hyp.index

        answer = microtests.model_image

        correct_label = answer.label

        is_correct = guess == correct_label

        correct += is_correct

        result = (num, answer.index, correct_label, guess_num, guess, is_correct)

        results.append(result)

        log.info(
            "Num %5s, Input image %5s, Expected label %s, Guessed image %5s, Guessed num %s. Correct: %5s, Accuracy %3.0f%%",
            num,
            answer.index,
            correct_label,
            guess_num,
            guess,
            is_correct,
            (correct * 100) / (num + 1),
        )

    with open(path, "w") as f:
        f.write("[\n    " + ",\n    ".join(json.dumps(row) for row in results) + "\n]")

    end = datetime.datetime.now()

    log.info(
        "Got %s out of %s correct. %3.0f%%",
        correct,
        len(test_pictures),
        (correct * 100) / len(test_pictures),
    )

    log.info("Did %s in %s", len(test_pictures), end - start)


def main():

    apps = {"experiment": experiment, "example": example}

    rng = random.Random()

    agent_count = 200

    max_iterations = 200

    try:
        app = sys.argv[1] in apps and sys.argv[1]
        f = apps[sys.argv[1]]
    except (KeyError, IndexError) as err:
        print(f"Usage: {sys.argv[0]} ({' | '.join(apps)})")
    else:

        log.info(
            "Starting %s. Agent count: %s, Max Iterations: %s",
            app,
            agent_count,
            max_iterations,
        )
        f(agent_count=agent_count, max_iterations=max_iterations, rng=rng)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s: %(message)s",
        style="%",
    )

    main()
