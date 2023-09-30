# Demosaicking

The goal of this exercise is to simulate demosaicking algorithm, which converts raw data read from CMOS sensor into digital RGB image. 

In most modern cameras CFA, which stands for *color filter array* is build in such a way, that a single pixel only registers one color, which create so called *mosaicking*. The colors are arranged in a pattern, which is usually specific for given camera (two examples are provided in the exercise). Then full color image is constructed using *demosaicking* algorithms, which could be implemented by using image interpolation for example.

### Sources
* [https://en.wikipedia.org/wiki/Demosaicing](https://en.wikipedia.org/wiki/Demosaicing)
* [https://en.wikipedia.org/wiki/Bayer](https://en.wikipedia.org/wiki/Bayer)
* [http://nagykrisztian.com/store/hirakawa.pdf](http://nagykrisztian.com/store/hirakawa.pdf)
* [https://ui.adsabs.harvard.edu/abs/2006JEI....15a3003C/abstract](https://ui.adsabs.harvard.edu/abs/2006JEI....15a3003C/abstract)
* [https://paperswithcode.com/task/demosaicking](https://paperswithcode.com/task/demosaicking)
* [https://www.semanticscholar.org/paper/Color-filter-array-recovery-using-a-threshold-based-Chang-Cheung/361502aad08b474fa7a399532608f855651a9cc4](https://www.semanticscholar.org/paper/Color-filter-array-recovery-using-a-threshold-based-Chang-Cheung/361502aad08b474fa7a399532608f855651a9cc4)
