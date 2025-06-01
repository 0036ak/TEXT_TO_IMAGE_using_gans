# Text_To_IMAGE_using_GANs

This project takes some **text** and uses **GANs (Generative Adversarial Networks)** to create an **image** that matches the text.

## 📌 What is this project?

This is a simple project that shows how a computer can **read text** (like "a red flower") and **draw an image** based on it.  
It uses **Deep Learning** and a special technique called **GANs** to do this.

## 🤖 What are GANs?

**GANs** are two smart computer programs:
- One tries to **create fake images**.
- The other tries to **check if the image is real or fake**.
They both **learn together**, and over time, the image quality gets better.

## 📂 Project Structure

Here are some important parts of the project:

- `model/`: contains the GAN model files
- `data/`: has the training data
- `train.py`: file used to train the GAN
- `generate.py`: used to create images from text
- `utils.py`: helper functions

## 🚀 How to run the project

### 1. Clone the repo

```bash
git clone https://github.com/0036ak/TEXT_TO_IMAGE_using_gans.git
cd TEXT_TO_IMAGE_using_gans
