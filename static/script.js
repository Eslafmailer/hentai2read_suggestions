// script.js

// Function to clear author filter
function clearAuthorFilter() {
    document.querySelectorAll('#author-filters input[type="radio"]').forEach(radio => {
        radio.checked = false;
    });
    filterBooks();
    updateFilterCounts();
}

// In your existing script where you set up event listeners, add:
document.getElementById('clear-authors').addEventListener('click', clearAuthorFilter);

// Function to filter options based on search input
function filterOptions(inputId, containerId) {
    const searchText = document.getElementById(inputId).value.toLowerCase();
    const items = document.querySelectorAll(`#${containerId} label`);

    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        if (text.includes(searchText)) {
            item.previousElementSibling.style.display = '';
            item.style.display = '';
            item.nextElementSibling.style.display = '';
            item.nextElementSibling.nextElementSibling.style.display = '';
        } else {
            item.previousElementSibling.style.display = 'none';
            item.style.display = 'none';
            item.nextElementSibling.style.display = 'none';
            item.nextElementSibling.nextElementSibling.style.display = 'none';
        }
    });
}

// Event listeners for category and tag search inputs
document.getElementById('category-search').addEventListener('input', () => filterOptions('category-search', 'category-filters'));
document.getElementById('tag-search').addEventListener('input', () => filterOptions('tag-search', 'tag-filters'));

// Function to create checkbox or radio button filters with count
function createFilterWithCount(items, containerId, isRadio = false) {
    const container = document.getElementById(containerId);
    container.innerHTML = ''; // Clear existing content

    items.forEach(item => {
        const input = document.createElement('input');
        input.type = isRadio ? 'radio' : 'checkbox';
        input.id = item;
        input.name = isRadio ? 'author-filter' : item; // Radio buttons need the same 'name' attribute
        input.value = item;

        const label = document.createElement('label');
        label.htmlFor = item;
        label.textContent = item;

        // Span for showing count
        const countSpan = document.createElement('span');
        countSpan.id = `count-${containerId}-${item}`;
        countSpan.style.marginLeft = '5px';

        container.appendChild(input);
        container.appendChild(label);
        container.appendChild(countSpan);
        container.appendChild(document.createElement('br'));

        // Attach event listener to each input
        input.addEventListener('change', () => {
            filterBooks();
        });
    });
    updateFilterCounts(books);
}

// Function to update filter counts
function updateFilterCounts(filteredBooks) {
    // Update counts for category filters
    document.querySelectorAll('#category-filters input').forEach(checkbox => {
        const count = filteredBooks.filter(book => book.categories.includes(checkbox.value)).length;
        document.getElementById(`count-category-filters-${checkbox.value}`).textContent = `(${count})`;
    });

    // Update counts for tag filters
    document.querySelectorAll('#tag-filters input').forEach(checkbox => {
        const count = filteredBooks.filter(book => book.tags.includes(checkbox.value)).length;
        document.getElementById(`count-tag-filters-${checkbox.value}`).textContent = `(${count})`;
    });

    // Update counts for author filters
    document.querySelectorAll('#author-filters input').forEach(checkbox => {
        const count = filteredBooks.filter(book => book.authors.includes(checkbox.value)).length;
        document.getElementById(`count-author-filters-${checkbox.value}`).textContent = `(${count})`;
    });
}

// Function to add filters
function addFilters() {
    const categories = new Set();
    const tags = new Set();
    const authorCounts = {};

    // Extracting unique categories, tags, and authors
    books.forEach(book => {
        book.categories.forEach(category => categories.add(category));
        book.tags.forEach(tag => tags.add(tag));
        book.authors.forEach(author => {
            authorCounts[author] = (authorCounts[author] || 0) + 1;
        });
    });

    const authors = Object.keys(authorCounts).sort((a, b) => authorCounts[b] - authorCounts[a]);

    // Creating checkbox filters with counts for categories, tags
    createFilterWithCount(categories, 'category-filters');
    createFilterWithCount(tags, 'tag-filters');

    // Creating radio button filters for authors
    createFilterWithCount(authors, 'author-filters', true);
}

// Function to filter books based on selected categories and tags
function filterBooks() {
    const selectedCategories = new Set();
    const selectedTags = new Set();
    const selectedAuthors = new Set();

    // Collecting selected categories
    document.querySelectorAll('#category-filters input:checked').forEach(checkbox => {
        selectedCategories.add(checkbox.value);
    });

    // Collecting selected tags
    document.querySelectorAll('#tag-filters input:checked').forEach(checkbox => {
        selectedTags.add(checkbox.value);
    });

    document.querySelectorAll('#author-filters input:checked').forEach(checkbox => {
        selectedAuthors.add(checkbox.value);
    });

    const filteredBooks = books.filter(book => {
        const categoryMatch = selectedCategories.size === 0 || Array.from(selectedCategories).every(category => book.categories.includes(category));
        const tagMatch = selectedTags.size === 0 || Array.from(selectedTags).every(tag => book.tags.includes(tag));
        const authorMatch = selectedAuthors.size === 0 || Array.from(selectedAuthors).every(author => book.authors.includes(author));
        return categoryMatch && tagMatch && authorMatch;
    });

    displayBooks(filteredBooks);
    updateFilterCounts(filteredBooks);
}

// Modified displayBooks function to accept an array of books
function displayBooks(filteredBooks) {
    const booksContainer = document.getElementById('books');
    booksContainer.innerHTML = '';

    const max = Math.min(50, filteredBooks.length);
    for (let ind = 0; ind < max; ind++) {
        const book = filteredBooks[ind]
        const bookElement = document.createElement('a');
        bookElement.target = "_blank";
        bookElement.href = book.href;
        bookElement.className = 'book';
        bookElement.innerHTML = `
            <img src="${book.coverUrl}" alt="${book.name}">
            <span>${book.name}</span>
        `;
        booksContainer.appendChild(bookElement);
    }
}


// Initial setup
addFilters();
displayBooks(books);
