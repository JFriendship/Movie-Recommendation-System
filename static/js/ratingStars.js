const range = document.getElementById("rating");
const starsFill = document.querySelector(".stars-fill");

function updateStars(value) {
    const percent = (value / 5) * 100;
    starsFill.style.width = percent + "%";
}

range.addEventListener("input", (e) => {
    updateStars(parseFloat(e.target.value));
});

updateStars(range.value);