const box = document.getElementById("search-box");
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
    `<li role="option" tabindex="0" class="search-suggestion-item">${item}</li>`
  ).join("");

  const options = results_div.querySelectorAll("li");
  options.forEach(option => {
    option.addEventListener("click", () => {
      box.value = option.textContent;
      results_div.innerHTML = "";
    });
    
    option.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        box.value = option.textContent;
        results_div.innerHTML = "";
      }
    });
  });
});

