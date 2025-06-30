import React from 'react';
import { Product } from '../types';

interface ProductCardProps {
  product: Product;
}

export const ProductCard: React.FC<ProductCardProps> = ({ product }) => {
  const formatPrice = (price?: number) => {
    return price ? `$${price.toLocaleString()}` : 'Price not available';
  };

  const getImageUrl = () => {
    return product.image_urls[0] || 'https://via.placeholder.com/280x200/f3f4f6/9ca3af?text=No+Image';
  };

  return (
    <div className="product-card">
      <img
        src={getImageUrl()}
        alt={product.product_name}
        className="product-image"
        onError={(e) => {
          const target = e.target as HTMLImageElement;
          target.src = 'https://via.placeholder.com/280x200/f3f4f6/9ca3af?text=No+Image';
        }}
      />
      
      <h3 className="product-name">
        {product.product_name}
      </h3>
      
      <p className="product-sku">SKU: {product.sku}</p>
      
      {product.key_features && (
        <p className="product-features">
          {product.key_features}
        </p>
      )}
      
      <div className="product-footer">
        <span className="product-price">
          {formatPrice(product.price)}
        </span>
        
        {product.size && (
          <span className="product-size">
            {product.size}"
          </span>
        )}
      </div>
      
      <a
        href={product.product_url}
        target="_blank"
        rel="noopener noreferrer"
        className="product-link"
      >
        View Product
      </a>
    </div>
  );
};