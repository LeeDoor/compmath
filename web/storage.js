const form = document.querySelector("form[data-storage-key]");
if (form) {
  const key = `compmath.${form.dataset.storageKey}.v2`;
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(key) || "{}");
  } catch {
    saved = {};
  }
  if (!saved || typeof saved !== "object") saved = {};

  for (const field of form.querySelectorAll("input[name], select[name]")) {
    if (Object.hasOwn(saved, field.name)) field.value = saved[field.name];
  }

  const save = () => {
    const values = {};
    for (const field of form.querySelectorAll("input[name], select[name]")) {
      values[field.name] = field.value;
    }
    localStorage.setItem(key, JSON.stringify(values));
  };
  form.addEventListener("input", save);
  form.addEventListener("change", save);
  form.addEventListener("submit", save);

  for (const select of form.querySelectorAll(
    'select[name="variables"], select[name="constraints"], ' +
      'select[name="warehouses"], select[name="customers"]',
  )) {
    select.addEventListener("change", () => {
      save();
      const query = new URLSearchParams(new FormData(form));
      window.location.assign(`${form.dataset.resizeUrl}?${query}`);
    });
  }
}
