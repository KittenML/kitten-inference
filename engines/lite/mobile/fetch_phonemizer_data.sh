#!/usr/bin/env sh
# Downloads the espeak-ng data files that the text front ends of the C library read, from espeak-ng's 1.52.0 release
# (the version the Python package runs through espeakng_loader), and checks their SHA-256 sums:
#   en          en_rules, en_list: Kitten TTS 0.8, English (kt08_phonemizer_load)
#   <language>  <lang>_rules, <lang>_list (+ it_listx, tr_listx, ru_listx), the language's ph_* file and the shared phoneme
#               tables (phsource/phonemes, ph_base2); also the English data (every language reads some English
#               words with the English rules), for Italian the German, Spanish and French data, and for
#               Portuguese the German data, for the foreign words their rules mark: Kitten TTS 2 mini / nano /
#               pico (espeak_data_dir of kt08_tiny_frontend_load)
#               languages: es it pt fr de hi ru (nano, pico, mini) and da fi ja ko nl sv sw tr (mini only)
#   ja          not espeak-ng: the UniDic dictionary (unidic-lite 1.0.8, BSD) and misaki's word list, into OUT/ja/
#
#   engines/lite/mobile/fetch_phonemizer_data.sh [OUT_DIR [LANGUAGE ...]]   (default: ./phonemizer-data, all)
#
# These files are espeak-ng's, under the GNU General Public License v3.0 (https://github.com/espeak-ng/espeak-ng);
# they are data, not code, and are not part of this repository or of the library. An app downloads them at run time
# (for example on first use, like the model weights) or lets the user fetch them, and passes their folder to the
# library. Apps that mirror the files should keep the same bytes (the sums below).
set -eu
OUT=${1:-phonemizer-data}
[ $# -gt 0 ] && shift
LANGS=${*:-"en es it pt fr de da fi hi ja ko nl ru sv sw tr"}
COMMIT=4870adfa25b1a32b4361592f1be8a40337c58d6c   # tag 1.52.0
BASE=https://raw.githubusercontent.com/espeak-ng/espeak-ng/$COMMIT
mkdir -p "$OUT/phsource"
sha() { (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -d' ' -f1; }
japanese() {   # UniDic (unidic-lite 1.0.8 from PyPI) + misaki's ja_words.txt -> OUT/ja/
  mkdir -p "$OUT/ja"
  JA_OK=1
  for f in "sys.dic 122c4c91f026bf4b65bbe2dfd8b9f8eeb6ab56d8eb2a7287a68bca75708ba513" \
           "unk.dic e3b92803feeb6c2712c796b905317ec192a59729d2c1e92b2728d454169541c4" \
           "matrix.bin 88fb99efe1075d9b8b40e01bb2751e4095f1612618f2490af776eb5ed39190a9" \
           "char.bin dd31396563d8924645b80fd3c9aa7b13ca089d7748f25553a1d6bc3f9b511ae8" \
           "BSD 7917ecfaf95a5ba7b4f45aa9a845ac931d79c90f39b7e711442dbc21c612560f" \
           "COPYING 178ffb7cbc2beadb0b87a14497e58222b484076ba7cfc8938a07ac7627fa22c4"; do
    set -- $f
    [ -f "$OUT/ja/$1" ] && [ "$(sha "$OUT/ja/$1")" = "$2" ] || JA_OK=0
  done
  if [ $JA_OK = 0 ]; then
    tgz="$OUT/ja/unidic-lite-1.0.8.tar.gz.part"
    curl -fsSL --retry 3 -o "$tgz" https://files.pythonhosted.org/packages/55/2b/8cf7514cb57d028abcef625afa847d60ff1ffbf0049c36b78faa7c35046f/unidic-lite-1.0.8.tar.gz
    got=$(sha "$tgz")
    if [ "$got" != db9d4572d9fdd4d00a97949d4b0741ec480ee05a7e7e2e32f547500dae27b245 ]; then
      rm -f "$tgz"; echo "fetch_phonemizer_data: unidic-lite-1.0.8.tar.gz has SHA-256 $got" >&2; exit 1
    fi
    P=unidic-lite-1.0.8/unidic_lite/dicdir
    for n in sys.dic unk.dic matrix.bin char.bin BSD COPYING; do
      tar -xzf "$tgz" -O "$P/$n" > "$OUT/ja/$n.part"
      mv "$OUT/ja/$n.part" "$OUT/ja/$n"
    done
    rm -f "$tgz"
    for f in "sys.dic 122c4c91f026bf4b65bbe2dfd8b9f8eeb6ab56d8eb2a7287a68bca75708ba513" \
             "unk.dic e3b92803feeb6c2712c796b905317ec192a59729d2c1e92b2728d454169541c4" \
             "matrix.bin 88fb99efe1075d9b8b40e01bb2751e4095f1612618f2490af776eb5ed39190a9" \
             "char.bin dd31396563d8924645b80fd3c9aa7b13ca089d7748f25553a1d6bc3f9b511ae8"; do
      set -- $f
      [ "$(sha "$OUT/ja/$1")" = "$2" ] || { echo "fetch_phonemizer_data: ja/$1 has the wrong SHA-256" >&2; exit 1; }
    done
  fi
  words="$(dirname "$0")/../../../src/kitten_inference/tts2_tiny/frontend/misaki/data/ja_words.txt"
  [ -f "$words" ] || { echo "fetch_phonemizer_data: $words not found (run the script from a kitten-inference checkout)" >&2; exit 1; }
  [ "$(sha "$words")" = a93a8e8aee24db307a32becb8bf01c4c2908ecf37e6c91f7a705fafdfeba67ff ] || { echo "fetch_phonemizer_data: $words has the wrong SHA-256" >&2; exit 1; }
  cp "$words" "$OUT/ja/ja_words.txt"
}
fetch() {  # dictsource/[extra/]<name> or phsource/<name>, sha256; dictsource files go to OUT/, phsource files to OUT/phsource/
  case $1 in dictsource/extra/*) dst="$OUT/${1#dictsource/extra/}" ;; dictsource/*) dst="$OUT/${1#dictsource/}" ;; *) dst="$OUT/$1" ;; esac
  if [ -f "$dst" ] && [ "$(sha "$dst")" = "$2" ]; then return 0; fi
  curl -fsSL --retry 3 -o "$dst.part" "$BASE/$1"
  got=$(sha "$dst.part")
  if [ "$got" != "$2" ]; then
    rm -f "$dst.part"
    echo "fetch_phonemizer_data: $1 has SHA-256 $got, expected $2" >&2
    exit 1
  fi
  mv "$dst.part" "$dst"
}
tables() {
  fetch phsource/phonemes 26e160620857ef986f677bd85ea077327669db087be2ef68e302e21c7c3801eb
  fetch phsource/ph_base2 203b364287ba9099a33e915cfe2d05b3dfd62722d2af2517a904d08b4600f244
}
english() {
  fetch dictsource/en_rules 884550dda08d0ba9d5ca713bf3a5ed295002d5de8a43b4346a81fcd040d19dc8
  fetch dictsource/en_list 78519730c462888dd9445042e48ebe07d461a1ed57ff70f1139f4cc9a732b9e0
  fetch phsource/ph_english cac553230cfab540e0a9589f393a53ac2ba5421f3112321b833736c0717b7e39
}
spanish() {
  tables
  fetch dictsource/es_rules a29d7c822c1f07c251f305bff09f52cd57bd8e52d5150b07054284f1103d04c9
  fetch dictsource/es_list 02ce8850eeea125d3fbd4b1985a35a341435571804a0b40b66d3c728a8697bed
  fetch phsource/ph_spanish cd266030a05040bcbc6bf2739fce701504426ce268ca2fd830902d1df591d311
}
german() {
  tables
  fetch dictsource/de_rules 72516bb26d68e8cea7032c741fe8446599e0b7f6eede06746e2c7a264fb789ed
  fetch dictsource/de_list fb01f4837d4022ec290bc471b3ad5c4de152f1c47ed00ac7e245738c545665e0
  fetch dictsource/de_emoji 666e91ab72ef2fc3471f0af6cc17cf6edb5325616f223bc0b2adba34cc959d41
  fetch phsource/ph_german 3d1e232ade65aadc2e71546ae804f6eb84d3cb0d32006c0c5cdbbd86943d464c
}
french() {
  tables
  fetch dictsource/fr_rules fa22d7e634dc63e63c9563ebba9937ad9083df2ca98f1bd0596c68aa75ebaaba
  fetch dictsource/fr_list b47e0fb49e62ea433de8e0c1b3b28a8843f5e1843c6236be1f6f1c551e7bf985
  fetch phsource/ph_french 49e1cf0e257867eceefa84fe49a4772ee8f62d9f61604f49cb8adbd385f1e419
}
danish() {
  tables
  fetch dictsource/da_rules a5a7ada6708ecb8efe202c7c7cdeda1a50799eddca6f4dd062f0abfc47197209
  fetch dictsource/da_list e284d1b5c48cce01fd6538af9f41759914673985af96982f659e7c941d1aa30e
  fetch phsource/ph_danish 2527a827d556a54a9b71f9346a51db7f1ad5845f7c5337cecc39918eafcf2769
}
portuguese() {
  tables
  fetch dictsource/pt_rules c1cbaad4b3eeda819bdda874704b963cf38f0021ec40e22764699fd7e066e006
  fetch dictsource/pt_list a447f522607875abfd00d0f8a4cc2d1e470b06e91838db3a46e51577676b940a
  fetch phsource/ph_pt_brazil 50bbedbc6ac16ac48c61dde23d358266604ccae515a23c27f6a65c6958742075
}
swahili() {
  tables
  fetch dictsource/sw_rules e9bc58baf9f0a1c6529610d78e6a2d434dea659c3afcadaa5f74f2f1a0c3fd1d
  fetch dictsource/sw_list c0216caac4fccaf3d1fad86ee12b265b525f92ee135ea95f79f945b1c8d57c6b
  fetch phsource/ph_swahili d8a8a01c6908ac6c6a8837e041bf4a435ca494ad3add904365fda6eeedcec31f
}
for l in $LANGS; do
  case $l in
    en) english ;;
    es) spanish
        english ;;
    it) tables
  fetch dictsource/it_rules c638c98c7e285104deeb8ad4cdcb6cb8eeeee2d509444849ed4beb310976691e
  fetch dictsource/it_list 83519af7fc2fe046e90eeb02594ba030e0510d73b8fd66ac7ddfece2bcd9802a
  fetch dictsource/it_listx 72c47e938b8eda3116209afb4bcc839e210eff9a3e3ebfc9127f3efe0e05793e
  fetch phsource/ph_italian bcd00fed107c3d0d2375de496b5bbf7e00dce74b2800f540ec68cf413d68df89
        english; german; spanish; french ;;   # words Italian rules read in those languages
    pt) portuguese
        english; german ;;                    # words the Portuguese list reads in those languages
    de) german
        english ;;                            # English words inside German text
    fr) french
        english ;;                            # names and loan words the French rules read as English
    da) danish
        english; spanish ;;                   # English words, and the words with n-tilde (_^_ES)
    sw) swahili ;;
    tr) tables
  fetch dictsource/tr_rules 08b7ca44468b5ade9e2293b0762b1ff7184930f8c5f24128ccb5ab37ecf02018
  fetch dictsource/tr_list acdce9f276664d421a57ed9dfa8b56854e3c1fd09c0d487aa0ad344d4cf9c180
  fetch dictsource/tr_listx 8c3f4508b960260a8bff0a4cd5f3b9ae716ce54d6ab020b6e590e8b5cafd3a9c
  fetch phsource/ph_turkish aa0d7b86dbfdbf31ac2d5174dbbf22fc821227608fb691c6d8029bcfe7ace37f
        ;;
    sv) tables
  fetch dictsource/sv_rules ce23bc2e1b2b72a1b46cc52619212fd24a9c91be6818df5322a6bab081361c10
  fetch dictsource/sv_list ce091b0678bf1a4c78ac8263a620327e56380babf0e75f20cd2ec95a70a25a42
  fetch phsource/ph_swedish 5f242a4eebac00df24159d30cc0fa1b1c7a1367f519d65b50f2b4014bb72f8b1
        english ;;
    fi) tables
  fetch dictsource/fi_rules deca068c6e46f029f0be872cb1efb8f69024aa19d0f964497e82c021f7ee74ff
  fetch dictsource/fi_list c4486c06f3ac57e4b76484d8b3cee5f98dc0a61d22f3307cbb5bc13f92aa3171
  fetch phsource/ph_finnish 8bea935bb6dfabd385a57198bf2e9b3437cff04c0a8be8338e673911d34f92ab
        ;;
    ko) tables
  fetch dictsource/ko_rules 3d3e9f15bcb1e7542f4e6a934359108ab78b09e24f4f2c50eaf80a1f70785555
  fetch dictsource/ko_list 13ae4784f7519b584445e296bd57b7e3481d1e3d4b54d952a3d02da857e2f43e
  fetch phsource/ph_korean 8afc766a864b0e464744713f626f6e53a831b831611215f05b62ae997a8fb891
        english ;;
    hi) tables
  fetch dictsource/hi_rules a21f45d5313fa01ee67179f7443734b2b55cf51857ac862afa7ed57fcb3b4fb5
  fetch dictsource/hi_list 31bb66481ad158058ed2e0816e4f2eb36783ecacb03a09555bd8d04bdf44922e
  fetch phsource/ph_hindi_base 265ddf6af15af6ee145b2cf73182ebedb8a9466e08845b53ffd8f66b23697d6a
  fetch phsource/ph_hindi 0f6a90e4b98bd660b34683f6938de1049fa590da875cd7a1459156215bf79386
  fetch phsource/ph_consonants 41e6ba5faba9fcd53b0cd1c336e354ed44528b33e430540fdcf30c0ec70f6344
        english ;;
    ru) tables
  fetch dictsource/ru_rules 62f3a8f3f7b94d99077de0decf7e8a8a3dc9ca653863eff5b717b6ae465e59af
  fetch dictsource/ru_list 714956c28278e3d776f08263a446b4e10735051aebbbf0227974309aa69eb02f
  fetch dictsource/extra/ru_listx fa511621c879954bcf69b0cb1a6af4c53cf9ccc5269fbd4928896b56ffecc6e2
  fetch phsource/ph_russian 97b6ddfa668b7db38b786bfbbb626ced061e65423e415fc9fdadc57d5175bf1d
  fetch phsource/ph_consonants 41e6ba5faba9fcd53b0cd1c336e354ed44528b33e430540fdcf30c0ec70f6344
  fetch phsource/ph_polish 43b9bdbf9cb4c8e65d7daf7f09d77f0b8cbd0b2d9c3c0054d5addb43004bd53a
        english ;;                            # ru_listx: the stress list (23 MB); Latin-letter words read as English
    nl) tables
  fetch dictsource/nl_rules 0152076c8dcf51e2c91562da2439d929a5e24d9a3ee2db451ea76e14256a74f2
  fetch dictsource/nl_list c74115dd1755b2bf3f8063b4fdb03345af33bfbf58253a3ff192f9adcada3a25
  fetch phsource/ph_dutch 7977911fb873c3435d6e95b80de6f3f96525e4970a81dbff94d161ae9b2c62c2
  fetch phsource/ph_afrikaans 0b4a22e6a57dee6fef7d0a90739b608ff30de51401ee9e134008fb1d47e4c721
  fetch phsource/ph_consonants 41e6ba5faba9fcd53b0cd1c336e354ed44528b33e430540fdcf30c0ec70f6344
  fetch phsource/ph_french 49e1cf0e257867eceefa84fe49a4772ee8f62d9f61604f49cb8adbd385f1e419
        english ;;                            # ph_dutch imports from the af, fr, en and consonants tables
    ja) japanese ;;
    *) echo "fetch_phonemizer_data: no data list for language '$l'" >&2; exit 2 ;;
  esac
done
echo "phonemizer data for: $LANGS in $OUT (espeak-ng $COMMIT = release 1.52.0, GPL-3.0; ja: UniDic 2.1.2, BSD)"
