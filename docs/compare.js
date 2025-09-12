// Simple before/after slider (works with .compare block)
document.querySelectorAll('.compare').forEach(cmp => {
  const before = cmp.querySelector('.before');
  const slider = cmp.querySelector('.slider');
  if (!before || !slider) return;

  const set = v => {
    const pct = Math.max(0, Math.min(100, Number(v)));
    before.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  };
  set(slider.value);
  slider.addEventListener('input', e => set(e.target.value));
});
