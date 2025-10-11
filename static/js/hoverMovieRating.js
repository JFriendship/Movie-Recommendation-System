document.querySelectorAll(".recommendations-list-item-wrapper").forEach(wrapper => {
  const range = wrapper.querySelector(".star-rating-range");
  const fill = wrapper.querySelector(".stars-fill");

  function updateStars() {
    const percent = (range.value / 5) * 100;
    fill.style.width = percent + "%";
  }

  // Update on input
  range.addEventListener("input", updateStars);

  wrapper.addEventListener('mouseleave', function() {
    range.value = 0.0;
    updateStars();
  });

  // Initialize
  updateStars();
});