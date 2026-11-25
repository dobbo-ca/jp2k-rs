//! Ports of `internal/jp2k/rounding_test.go`: the output of the default, `separate_rounding` and
//! `jj2000` 9/7 decodes of the fixtures, pinned by SHA-256. `separate_rounding` was checked
//! against OpenJPEG 2.4.0 built -O2 for x86-64, so CI catches a change without that build.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_memcpy
)]
mod common;

use common::*;
use jp2k::Options;

const SEPARATE: &[(&str, &[&str])] = &[
    (
        "fl_dense.j2k",
        &[
            "b84c84ab5133c4adf0718fc7fe8c03d49d48887f4607af6aeff4384398c057f0",
            "0055ff90459b4048df9ab081f9c1225483652bce93090db2cd2dec797994fae2",
            "380297cc27c74694d2e4604fe120190763cbd537abfb6fb7fe2cc340adfb3426",
        ],
    ),
    (
        "fl_sparse.j2k",
        &[
            "07fd2c130686768853fca81751d355ca0e3327d04e6ce5a27f648ef607bc7198",
            "d6494c9c69c3278813b1a79365798a048907ceada2e2c332561a9446279ec600",
            "3e31c6b4ad04deff6e7ea26a5e4c4a266c182201be6e33890f0b21e640d5ef1a",
        ],
    ),
    (
        "fl_empty.j2k",
        &[
            "bdb4fa7759ffe13eda3aea0bf150ffe29e3d942148977c4ab3f33f20037f7512",
            "a4c89fc02700a283b7a6697611de7ec260f4a52ea5be0ab9fc07a76b9fd196d8",
            "1ed9dc572147859f97b1226f695cad70bda38de97106167b6e5b9ec6c2377931",
        ],
    ),
    (
        "gray16x3_97_nomct.j2k",
        &[
            "4df91bb5ab4b61d1513a933aac1033601d0d9eecdc0603eda4c6690fb20c49b2",
            "18b79c93279b0ffe9a13140406b2348d038715f243713e0af8af697dc7db335a",
            "446cd7c119aee311cd286c381f3be57b9d73afea8bce0fe550fed1deb09aaebe",
        ],
    ),
    (
        "rgb8_ict_m5.j2k",
        &[
            "06ca68445e9fd211118ca8ab2c29b0f3d07e02cb2047f7ab74ae486eb8a3bac4",
            "4b595aa2ae18047648d1cc1366cc814d96a2e6bc05fb382cf98bcb349b8ce017",
            "6e2e2e94321086b67ba479b9e0445badd3c4fcb437a1ffb2a3b63c7da0e282c4",
        ],
    ),
    (
        "g48x40_3c16_97.j2k",
        &[
            "14c5807710daceb0c67d539351242cf2d768170ddf1aaf4fd2e8cfde3eb74ec4",
            "36a9d998a095b6a16aa645bd01d391fd6507be50fd9ed0dc5926ca4909d9a3fc",
            "03cc7815d99682f0265e9149d4391c4368a863b2b6aef054cc778f1a5b93c277",
        ],
    ),
    (
        "g96_3c16_97.j2k",
        &[
            "1c454e8529b90cd7933768b5e8d087ee0a41947c5d6450b258b9f58b3cbdf7ec",
            "3e7af37ca0bedf21308d95b6e457f65c86268319e52c6265aeba6ab83c2d9116",
            "8bfa91df433fc7e227985ad404d253b14ce7c41f8840dbb4ee1549fc79edecf9",
        ],
    ),
    (
        "svs256.j2k",
        &[
            "9c22209f5840e42de00c8796a4bf2749ea6973808a194a36ba127fc6eaeb41ab",
            "7c7badcfb8499da4b76797e802b986c58cde89ae13a9cd56e34d99be35ec1d74",
            "00ac12d9346a3b0e721a789bad900c4f083b1d2f80c6cf8a93d843dca234c7af",
        ],
    ),
    (
        "vsi512_dense.j2k",
        &["bb17e1137ba16e4e222e7c8c9622fae1edadf7ed7e453c63c7fb78a5da12f21f"],
    ),
    (
        "vsi512_background.j2k",
        &["807805275cb025da61f209d231ecab4fa7df892425b171f2f526ccddf4bc574f"],
    ),
    (
        "g64_97.j2k",
        &["6802927ae64595c258ad82a1b62fca35188361c04b4e3c3229748080fafac25c"],
    ),
    (
        "t81_r40_97.j2k",
        &["e98132042d77448a46f90ae110b9d8f281f5316e8be8cbfcfccde8272fc599a4"],
    ),
];

const DEFAULT: &[(&str, &[&str])] = &[
    (
        "fl_dense.j2k",
        &[
            "be516e956f99f361986c43806620d410c965faa0bd6f9e8975a748a527e8aabf",
            "cc4830b4ad09fa5a8f3948fd2674ecf6fa4b5b3e876655daa2d14ab4e1b8462f",
            "d64cf15e857bd700b1bb2f230e0d75b33e3e6d90c8bafc680783708363a72d54",
        ],
    ),
    (
        "fl_sparse.j2k",
        &[
            "8ecf2724cf48d97a222b52b875314dcf80b845134597d32f962e304cc3dccb9f",
            "8bf8b5cccde33fefef1bcd343eef1db659afb85fc73a54bb5a6e8dcdb3690ebb",
            "2e5b3cd676ed84b4e583dd279c5140ca78d6c86b172af50d7b81da22bd498fc6",
        ],
    ),
    (
        "fl_empty.j2k",
        &[
            "f13651768e904c4fa2efeeb77e162213c99d140fe1866cd2fb41cc62c7e8c29e",
            "c3789654f3afbb98ce04f62d1ba1546b7149e0ddd82fbe73a2d88f414e7972bb",
            "7b225a4bd0ece0159eda375f08ff9d30a8fc55b233f7af6f959ae4dd9e031fb5",
        ],
    ),
    (
        "gray16x3_97_nomct.j2k",
        &[
            "bd5260f0b3dbe04b462c878285278c4f116173f503c731c54475412b6e22ae8c",
            "b5ca219e3e3ad18e1f4567959f5c5d31f4633f6f82cae626ac157af5784f7be2",
            "e9e2aa8409244d6a1f989b99876900bbde91465ca19f74a994ccb1bacf06aa66",
        ],
    ),
    (
        "rgb8_ict_m5.j2k",
        &[
            "06ca68445e9fd211118ca8ab2c29b0f3d07e02cb2047f7ab74ae486eb8a3bac4",
            "4b595aa2ae18047648d1cc1366cc814d96a2e6bc05fb382cf98bcb349b8ce017",
            "6e2e2e94321086b67ba479b9e0445badd3c4fcb437a1ffb2a3b63c7da0e282c4",
        ],
    ),
    (
        "g48x40_3c16_97.j2k",
        &[
            "14c5807710daceb0c67d539351242cf2d768170ddf1aaf4fd2e8cfde3eb74ec4",
            "36a9d998a095b6a16aa645bd01d391fd6507be50fd9ed0dc5926ca4909d9a3fc",
            "03cc7815d99682f0265e9149d4391c4368a863b2b6aef054cc778f1a5b93c277",
        ],
    ),
    (
        "g96_3c16_97.j2k",
        &[
            "66319622ead2f1d9b87c4a8f58450996847bee3555598a6096819f46415b825f",
            "f4df99c7c04015fec3e4131185a2cdfb1a62256be33521eead5cc49916878401",
            "d79e52c3e8c874f4efebebe5b09d19741ea7c01dd4dbee807b32bd12e7f81ff4",
        ],
    ),
    (
        "svs256.j2k",
        &[
            "be604c05e8c85cdb511e123278315e6533601018098c8476af3eefb6e922f79f",
            "47461c6fdbec041ed27b64fa7e6571f703ab7c385d985485ad456e22d12a1e39",
            "7d68b4a609620b9f70e2fb83a87953582c5028a516c0b581f8284abd46af46aa",
        ],
    ),
    (
        "vsi512_dense.j2k",
        &["85f736ff8a2dd67d75b52e9ced97f5f4085dcbeca0be0c33edcea97b5097aa3d"],
    ),
    (
        "vsi512_background.j2k",
        &["62bb962f5246c869f32da7d068eeee078e222aac8db89ef0fdf95fdb4c4c4ee4"],
    ),
    (
        "g64_97.j2k",
        &["6802927ae64595c258ad82a1b62fca35188361c04b4e3c3229748080fafac25c"],
    ),
    (
        "t81_r40_97.j2k",
        &["e98132042d77448a46f90ae110b9d8f281f5316e8be8cbfcfccde8272fc599a4"],
    ),
];

const JJ2000: &[(&str, &[&str])] = &[
    (
        "fl_dense.j2k",
        &[
            "b78a9bed24874e19b7983ce4dc435492c58ff28da8dacf8a34094ee31b7bd6a2",
            "90c2aa08e604eeccd18ffdc140ab078a5a083bdd712b6c27e4ae8b30b9673cba",
            "d9d405d36728dcac96731d009279578076b8edc6f64930c54489dbd572d34b96",
        ],
    ),
    (
        "fl_sparse.j2k",
        &[
            "23801246d28e76882d1a8d96558f4e2ad9c855b57faa47835d41166a9bf0f5fc",
            "6aacb77ab990d0513fb427eb0084e5703e30a1d95f2a816cff41a315bd55a4d7",
            "c2b0e008ac290a0b785e5aea69419e8503ec170a4ca76237300af14b4d91a3ff",
        ],
    ),
    (
        "fl_empty.j2k",
        &[
            "7fdde50bdffc2402b2acf89389d8107e02a1a3eee9a1f690fcaaf5d452f934a6",
            "6be2d753f86d6bbc3c35868903d9a57fcffdf1fe943ade15bddd07638eeb9b6c",
            "be808f5b989691918adf0a0e5abefbf80a5613486790607fbd42d0c1f528e7e5",
        ],
    ),
    (
        "gray16x3_97_nomct.j2k",
        &[
            "2b7986875fc22734f2a3c75e13fe0600d4f790e6bbe76f2132bb861b282a5a52",
            "e1f1d1d65e8dde70a567da835be983b38fa1493930066353d36a077e5a045e41",
            "139fc7412475ad29004d105b3a336753c448882cc318ba78779791e27376292a",
        ],
    ),
    (
        "rgb8_ict_m5.j2k",
        &[
            "06ca68445e9fd211118ca8ab2c29b0f3d07e02cb2047f7ab74ae486eb8a3bac4",
            "03f6dc9b7d95db9090821ab527bcfa3f1b78bbaa4a338b3b3d13a85244f54ca7",
            "6e2e2e94321086b67ba479b9e0445badd3c4fcb437a1ffb2a3b63c7da0e282c4",
        ],
    ),
    (
        "g48x40_3c16_97.j2k",
        &[
            "011e295b9c7bec2a4174f7e8db0be402b56ad2632ed6fa9c13b94cef43fb1be8",
            "1e192f294189eef19f64b486e45697f458272ff3953439ae4e0067ddc210647a",
            "438ffedeecee261da44c9b976d5a69abe9980e85125c343e7d111f44479edde4",
        ],
    ),
    (
        "g96_3c16_97.j2k",
        &[
            "67e5f923fa851b004de1e2107d7f1858d219a198811bcb3fee14451a751d4552",
            "73e69d739133a9c9643a811d69af41a3d20b2d87d08f647031f2bfe47192d5b2",
            "367ec4716045687e8f163683abb6db3fd52e2277c33913b9fb3fb31ac080f1b6",
        ],
    ),
    (
        "svs256.j2k",
        &[
            "acf773f815c872385381e9943f99e2527df3a0e4bf116336b47786dbf156fe1e",
            "126a40b67d41318b6e4aee7390ed81de3ff7672062179f5ce45ad58dcc0e6a7e",
            "6966d7525f528e958aafc5dcafb460c76398104065335ba60384d95436ec3f40",
        ],
    ),
    (
        "vsi512_dense.j2k",
        &["01847cecc5a5b23567c524967a79cdf247212ccd832527995480c7ddc0c2c783"],
    ),
    (
        "vsi512_background.j2k",
        &["a678e8b94419e52cdc4d059ed058176747d865823624fdd05216d26e159438e1"],
    ),
    (
        "g64_97.j2k",
        &["4dc391e18cac636e88cc9fe7b5099144a24fcb314a689e7d4296e05fdc62085f"],
    ),
    (
        "t81_r40_97.j2k",
        &["bfa4b9dcd3e950be6e2dc4ea35d3c411faa99b57af894ba4d1c207c8ccaa8b5c"],
    ),
];

/// The default decode is checked against ICT/lift rounding by hash; the pgx bytes of each
/// component plane are `Gray16` bytes, big-endian.
fn plane_bytes(file: &str, n: usize, o: &Options) -> Vec<Vec<u8>> {
    let data = read(file);
    let comps: Vec<usize> = (0..n).collect();
    let planes =
        jp2k::decode_components(&data, &comps, o, &nc()).unwrap_or_else(|e| panic!("{file}: {e}"));
    planes
        .iter()
        .map(|p| p.pix.iter().flat_map(|v| v.to_be_bytes()).collect())
        .collect()
}

fn check_hashes(table: &[(&str, &[&str])], o: &Options, what: &str) {
    let mut bad = Vec::new();
    for (file, hashes) in table {
        for (c, pix) in plane_bytes(file, hashes.len(), o).iter().enumerate() {
            let got = sha256_hex(pix);
            if got != hashes[c] {
                bad.push(format!(
                    "{file} component {c}: sha256 {got}, want {}",
                    hashes[c]
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{what}: sha256 mismatch:\n{}",
        bad.join("\n")
    );
}

#[test]
fn default_fixtures() {
    check_hashes(DEFAULT, &Options::default(), "default");
}

#[test]
fn separate_rounding_fixtures() {
    let o = Options {
        separate_rounding: true,
        ..Options::default()
    };
    check_hashes(SEPARATE, &o, "separate rounding");
    // SeparateRounding differs from the default on the dense fixture.
    let sep = plane_bytes("fl_dense.j2k", 3, &o);
    let def = plane_bytes("fl_dense.j2k", 3, &Options::default());
    assert_ne!(sep, def, "separate rounding output equals the default");
}

/// JJ2000's output is pinned, and stays within 4 levels of `separate_rounding`: a lost or
/// doubled subband gain is a factor of 2.
#[test]
fn jj2000_fixtures() {
    let o = Options {
        jj2000: true,
        ..Options::default()
    };
    check_hashes(JJ2000, &o, "jj2000");
    let sep_o = Options {
        separate_rounding: true,
        ..Options::default()
    };
    for (file, hashes) in JJ2000 {
        let jj = plane_bytes(file, hashes.len(), &o);
        let sep = plane_bytes(file, hashes.len(), &sep_o);
        let mut same = true;
        for c in 0..jj.len() {
            for i in (0..jj[c].len() - 1).step_by(2) {
                let a = ((jj[c][i] as i32) << 8) | jj[c][i + 1] as i32;
                let b = ((sep[c][i] as i32) << 8) | sep[c][i + 1] as i32;
                assert!(
                    (a - b).abs() <= 4,
                    "{file} component {c} sample {}: JJ2000 {a}, separate {b}",
                    i / 2
                );
                same = same && a == b;
            }
        }
        if *file == "vsi512_dense.j2k" {
            assert!(!same, "{file}: JJ2000 output equals separate rounding");
        }
    }
}
