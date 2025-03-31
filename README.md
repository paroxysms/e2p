optimized, current benchmark is 0.3s for a 1080x720 image, including saving, but it uses deprecated functions due to laziness to understand how to use cowarray
ive reached my target, but i think a better benchmark is most certainly possible with decent optimizations
my last commits will be optimizations or refactoring to remove dependency on deprecated functions

this is just a Rust recode of the equivalent project written in python, it can be found [here](https://github.com/fuenwang/Equirec2Perspec)
my target benchmark time is also taken from the source above
