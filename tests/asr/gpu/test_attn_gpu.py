"""The Triton attention kernels on an NVIDIA GPU, without a model: the decode kernel that cuts each sequence's keys
into pieces gives the result of the single-pass decode kernel and of plain PyTorch attention."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA GPU", allow_module_level=True)

from kitten_inference.asr import _attn  # noqa: E402

if not _attn.probe("triton")[0]:
    pytest.skip("Triton cannot run here", allow_module_level=True)

NH, NKV, HD, PAGE = 16, 8, 128, 16


def _case(lens, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    B = len(lens)
    pages = [-(-n // PAGE) for n in lens]
    P = sum(pages) + 1
    K = torch.randn(P, PAGE, NKV, HD, device="cuda", generator=g).to(torch.bfloat16)
    V = torch.randn(P, PAGE, NKV, HD, device="cuda", generator=g).to(torch.bfloat16)
    bt = torch.zeros(B, max(pages), dtype=torch.int32, device="cuda")
    perm = torch.randperm(P - 1, generator=torch.Generator().manual_seed(seed)) + 1     # pages out of order
    i = 0
    for b, n in enumerate(pages):
        bt[b, :n] = perm[i:i + n].to(torch.int32)
        i += n
    q = torch.randn(B, NH, HD, device="cuda", generator=g).to(torch.bfloat16)
    cu = torch.arange(B + 1, dtype=torch.int32, device="cuda")
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return q, K, V, cu, seq, bt


@pytest.mark.parametrize("lens", [[1], [7], [300], [2047], [5, 900, 64, 1500], [17] * 16])
@pytest.mark.parametrize("splits", [2, 4, 8])
def test_split_decode_matches_single_pass_and_pytorch(monkeypatch, lens, splits):
    q, K, V, cu, seq, bt = _case(lens)
    max_k = bt.shape[1] * PAGE
    fn = _attn.make("triton", page=PAGE, nh=NH, nkv=NKV, hd=HD, max_seq=2048, device="cuda", dtype="bf16")
    ref_fn = _attn.make("torch", page=PAGE, nh=NH, nkv=NKV, hd=HD, max_seq=2048, device="cuda", dtype="bf16")
    monkeypatch.setattr(_attn, "DECODE_SPLITS", ())
    one = fn(q, K, V, cu, seq, bt, 1, max_k).float()
    monkeypatch.setattr(_attn, "DECODE_SPLITS", ((64, splits),))
    split = fn(q, K, V, cu, seq, bt, 1, max_k).float()
    ref = ref_fn(q, K, V, cu, seq, bt, 1, max_k).float()
    assert (split - one).abs().max().item() < 2e-2
    assert (split - ref).abs().max().item() < 2e-2
