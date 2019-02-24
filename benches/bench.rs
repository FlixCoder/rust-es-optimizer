#![feature(test)]
#![allow(dead_code, unused_variables, unused_imports)]

extern crate test;
extern crate esopt;
extern crate rand;

use test::Bencher;
use esopt::*;
use rand::prelude::*;


#[bench]
fn optimize(b:&mut Bencher)
{
    
    b.iter(||
        {
            gen_rnd_vec(500, 0.2)
        });
}

