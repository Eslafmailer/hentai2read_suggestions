// script.js

// Function to create checkbox filters with count
function createCheckboxFilterWithCount(items, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = ''; // Clear existing content

    items.forEach(item => {
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = item;
        checkbox.name = item;
        checkbox.value = item;

        const label = document.createElement('label');
        label.htmlFor = item;
        label.textContent = item;

        // Span for showing count
        const countSpan = document.createElement('span');
        countSpan.id = `count-${containerId}-${item}`;
        countSpan.style.marginLeft = '5px';

        container.appendChild(checkbox);
        container.appendChild(label);
        container.appendChild(countSpan);
        container.appendChild(document.createElement('br'));

        // Attach event listener to each checkbox
        checkbox.addEventListener('change', () => {
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

    // Creating checkboxes for categories and tags
    createCheckboxFilterWithCount(categories, 'category-filters');
    createCheckboxFilterWithCount(tags, 'tag-filters');
    createCheckboxFilterWithCount(authors, 'author-filters');}

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
