#!/usr/bin/env sh
# Downloads the espeak-ng data files that the text front ends of the C library read, at a pinned espeak-ng commit,
# and checks their SHA-256 sums:
#   en          en_rules, en_list: Kitten TTS 0.8, English (kt08_phonemizer_load)
#   es, it      <lang>_rules, <lang>_list (+ it_listx) and the phoneme definitions they use (phsource/phonemes,
#               ph_base2 and the language's ph_* file; also the English data, and for Italian the German, Spanish
#               and French data, for the foreign words its rules mark): Kitten TTS 2 mini / nano / pico
#               (espeak_data_dir of kt08_tiny_frontend_load)
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
LANGS=${*:-"en es it"}
COMMIT=59eb19938f12e30881c81d86ce4a7de25414c9f4
BASE=https://raw.githubusercontent.com/espeak-ng/espeak-ng/$COMMIT
mkdir -p "$OUT/phsource"
fetch() {  # dictsource/<name> or phsource/<name>, sha256; dictsource files go to OUT/, phsource files to OUT/phsource/
  case $1 in dictsource/*) dst="$OUT/${1#dictsource/}" ;; *) dst="$OUT/$1" ;; esac
  if [ -f "$dst" ] && [ "$( (sha256sum "$dst" 2>/dev/null || shasum -a 256 "$dst") | cut -d' ' -f1)" = "$2" ]; then return 0; fi
  curl -fsSL --retry 3 -o "$dst.part" "$BASE/$1"
  got=$( (sha256sum "$dst.part" 2>/dev/null || shasum -a 256 "$dst.part") | cut -d' ' -f1)
  if [ "$got" != "$2" ]; then
    rm -f "$dst.part"
    echo "fetch_phonemizer_data: $1 has SHA-256 $got, expected $2" >&2
    exit 1
  fi
  mv "$dst.part" "$dst"
}
english() {   # es and it read some English words with the English rules (espeak's _^_EN)
  fetch dictsource/en_rules 8e75e9341ea735cc514b29a7d3a95c6c241c1cc176ad43e5699b8f7f66ab3194
  fetch dictsource/en_list 24eb79018ed6253c10682096de672ce9265c1fe15c3e19e7f754d57a0fcd9790
  fetch phsource/ph_english cac553230cfab540e0a9589f393a53ac2ba5421f3112321b833736c0717b7e39
}
spanish() {
  tables
  fetch dictsource/es_rules a29d7c822c1f07c251f305bff09f52cd57bd8e52d5150b07054284f1103d04c9
  fetch dictsource/es_list ca1a24ffae70f8c3dd73df6a7c45c6001f98426f352d08047962de056b3e34dd
  fetch phsource/ph_spanish cd266030a05040bcbc6bf2739fce701504426ce268ca2fd830902d1df591d311
}
german() {   # only for the German words inside Italian text (_^_DE)
  tables
  fetch dictsource/de_rules 72516bb26d68e8cea7032c741fe8446599e0b7f6eede06746e2c7a264fb789ed
  fetch dictsource/de_list fb01f4837d4022ec290bc471b3ad5c4de152f1c47ed00ac7e245738c545665e0
  fetch phsource/ph_german 3d1e232ade65aadc2e71546ae804f6eb84d3cb0d32006c0c5cdbbd86943d464c
}
french() {   # only for the French words inside Italian text (_^_FR)
  fetch dictsource/fr_rules bd572b7595c48ca171b7a7eaf50a67bdfe24c8c3b9f2697c799833b97369349b
  fetch dictsource/fr_list b47e0fb49e62ea433de8e0c1b3b28a8843f5e1843c6236be1f6f1c551e7bf985
  fetch phsource/ph_french 49e1cf0e257867eceefa84fe49a4772ee8f62d9f61604f49cb8adbd385f1e419
}
tables() {
  fetch phsource/phonemes 332e5dd002234f4291bc71c69f400e14672a60a8f037461ffe82472f6cad23ba
  fetch phsource/ph_base2 203b364287ba9099a33e915cfe2d05b3dfd62722d2af2517a904d08b4600f244
}
for l in $LANGS; do
  case $l in
    en) fetch dictsource/en_rules 8e75e9341ea735cc514b29a7d3a95c6c241c1cc176ad43e5699b8f7f66ab3194
        fetch dictsource/en_list 24eb79018ed6253c10682096de672ce9265c1fe15c3e19e7f754d57a0fcd9790 ;;
    es) spanish
        english ;;
    it) tables
        fetch dictsource/it_rules c638c98c7e285104deeb8ad4cdcb6cb8eeeee2d509444849ed4beb310976691e
        fetch dictsource/it_list 83519af7fc2fe046e90eeb02594ba030e0510d73b8fd66ac7ddfece2bcd9802a
        fetch dictsource/it_listx 72c47e938b8eda3116209afb4bcc839e210eff9a3e3ebfc9127f3efe0e05793e
        fetch phsource/ph_italian bcd00fed107c3d0d2375de496b5bbf7e00dce74b2800f540ec68cf413d68df89
        english; german; spanish; french ;;   # words Italian rules read in those languages
    *) echo "fetch_phonemizer_data: no data list for language '$l' (en es it)" >&2; exit 2 ;;
  esac
done
echo "espeak-ng data ($COMMIT, GPL-3.0) for: $LANGS in $OUT"
