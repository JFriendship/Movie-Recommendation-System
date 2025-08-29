const box = document.getElementById("searchBox");
const results_div = document.getElementById("results");

box.addEventListener("input", async () => {
  const query = box.value;
  if (!query) {
    results_div.innerHTML = "";
    return;
  }

  const res = await fetch(`/search?q=${encodeURIComponent(query)}`);
  const data = await res.json();

  results_div.innerHTML = data.map(item => 
    `<div class="search-suggestion-item">${item}</div>`
  ).join("");
});