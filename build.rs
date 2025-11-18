/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

fn main() {
    println!("cargo:rerun-if-changed=src/stb_image.c");

    bindgen::Builder::default()
        .header("src/stb_image.c")
        .layout_tests(false)
        .allowlist_function("stbi_.*")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("src/stb_image.rs")
        .expect("Couldn't write bindings!");

    let mut build = cc::Build::new();

    build.cpp(true);
    build.define("STB_IMAGE_IMPLEMENTATION", None);
    build.file("src/stb_image.c");
    build.compile("libstb_image");
}
