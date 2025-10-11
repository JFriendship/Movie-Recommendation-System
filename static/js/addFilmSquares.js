let pageWidth = window.innerWidth;

window.addEventListener('resize', function() {
  pageWidth = window.innerWidth; 
  createFilmDeco()
});

function createFilmDeco() {
    const containers = document.querySelectorAll('#film-deco-wrapper');

    containers.forEach(container => {
        pageWidth = container.clientWidth;
        container.innerHTML = '';
        for (let i  = 0; i < (pageWidth / 36); i++) {
            const newDiv = document.createElement('div');
            newDiv.className = 'film-deco';

            container.appendChild(newDiv);
        }
    });
}

createFilmDeco();