document.querySelectorAll('.code-block').forEach(block => {
  if (!navigator.clipboard) return;
  const button = document.createElement('button');
  const french = document.documentElement.lang === 'fr';
  button.type = 'button';
  button.className = 'copy';
  button.textContent = french ? 'Copier' : 'Copy';
  button.setAttribute('aria-label', french ? 'Copier le code' : 'Copy code');
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(block.querySelector('code').textContent);
      document.getElementById('copy-status').textContent = french ? 'Code copié.' : 'Code copied.';
      button.textContent = french ? 'Copié' : 'Copied';
    } catch {
      document.getElementById('copy-status').textContent = french ? 'Sélectionnez le code pour le copier.' : 'Select the code to copy it.';
    }
  });
  block.appendChild(button);
});
